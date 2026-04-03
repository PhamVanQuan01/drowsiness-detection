from __future__ import annotations

from pathlib import Path
from collections import deque
import tempfile
import time
import base64
import json

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.alarm import AlarmPlayer
from src.ear import compute_average_ear, ear_to_closed_score
from src.face_mesh_utils import FaceMeshDetector
from src.inference import EyeStateClassifier


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "eye_state_model_final.keras"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map.json"
ALARM_PATH = BASE_DIR / "assets" / "alarm.wav"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_METADATA_PATH = DATA_DIR / "sample_metadata.csv"
EVALUATION_METRICS_PATH = DATA_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = DATA_DIR / "confusion_matrix.csv"
TRAINING_HISTORY_PATH = DATA_DIR / "training_history.csv"


# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="Drowsiness Detection",
    layout="wide",
    page_icon="🚗",
)


# =========================
# Cached resources
# =========================
@st.cache_resource
def load_detector() -> FaceMeshDetector:
    return FaceMeshDetector(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        eye_crop_padding_x=0.18,
        eye_crop_padding_y=0.30,
    )


@st.cache_resource
def load_classifier(model_path: str, label_map_path: str) -> EyeStateClassifier:
    return EyeStateClassifier(
        model_path=model_path,
        label_map_path=label_map_path,
        input_size=(224, 224),
        closed_label_name="Closed_Eyes",
    )


@st.cache_data
def load_metadata_df() -> pd.DataFrame | None:
    if not SAMPLE_METADATA_PATH.exists():
        return None
    return pd.read_csv(SAMPLE_METADATA_PATH)


@st.cache_data
def load_metrics_dict() -> dict | None:
    if not EVALUATION_METRICS_PATH.exists():
        return None
    with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_confusion_matrix_df() -> pd.DataFrame | None:
    if not CONFUSION_MATRIX_PATH.exists():
        return None
    return pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)


@st.cache_data
def load_training_history_df() -> pd.DataFrame | None:
    if not TRAINING_HISTORY_PATH.exists():
        return None
    return pd.read_csv(TRAINING_HISTORY_PATH)


# =========================
# Helper functions
# =========================
def classify_eye_state(classifier: EyeStateClassifier, eye_crop: np.ndarray | None) -> dict:
    if eye_crop is None or eye_crop.size == 0:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "probs": {},
        }
    return classifier.predict(eye_crop)



def fuse_eye_scores(
    avg_ear: float,
    left_closed_prob: float,
    right_closed_prob: float,
    alpha: float = 0.6,
    ear_open: float = 0.30,
    ear_closed: float = 0.18,
) -> tuple[float, float, float]:
    ear_score = ear_to_closed_score(
        ear=avg_ear,
        ear_open=ear_open,
        ear_closed=ear_closed,
    )
    cnn_score = (left_closed_prob + right_closed_prob) / 2.0
    fused_score = alpha * cnn_score + (1.0 - alpha) * ear_score
    return float(ear_score), float(cnn_score), float(fused_score)



def file_to_base64(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")



def render_browser_alarm(
    placeholder,
    alarm_wav_path: str | Path,
    should_play: bool,
    key: str = "browser_alarm",
) -> None:
    if not should_play:
        placeholder.empty()
        return

    audio_b64 = file_to_base64(alarm_wav_path)
    if not audio_b64:
        placeholder.warning("Không tìm thấy file alarm.wav để phát cảnh báo.")
        return

    audio_html = f"""
    <audio id="{key}" autoplay loop>
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
    </audio>
    <script>
        const audio = document.getElementById("{key}");
        if (audio) {{
            audio.play().catch((e) => {{
                console.log("Autoplay bị chặn:", e);
            }});
        }}
    </script>
    """
    placeholder.markdown(audio_html, unsafe_allow_html=True)



def draw_header_panel(frame: np.ndarray, panel_height_ratio: float = 0.18) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = int(h * panel_height_ratio)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)



def draw_status_ui(
    frame: np.ndarray,
    status_text: str,
    status_color: tuple[int, int, int],
    fps: float,
    avg_ear: float,
    ear_score: float,
    cnn_score: float,
    fused_score: float,
    drowsy_counter: int,
    recovery_counter: int,
    no_face_counter: int,
    drowsy_threshold_frames: int,
    recovery_threshold_frames: int,
    alert_on: bool,
) -> np.ndarray:
    display = draw_header_panel(frame)
    h, w = display.shape[:2]

    font_scale = max(w / 1100.0, 0.55)
    thickness = max(int(font_scale * 2), 1)

    cv2.putText(
        display,
        status_text,
        (int(w * 0.03), int(h * 0.07)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        status_color,
        thickness + 1,
        cv2.LINE_AA,
    )

    if alert_on:
        cv2.rectangle(display, (10, 10), (w - 10, h - 10), (0, 0, 255), 3)
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(
                display,
                "DROWSY ALERT!",
                (int(w * 0.03), int(h * 0.15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.95,
                (0, 0, 255),
                thickness + 1,
                cv2.LINE_AA,
            )

    bar_x1 = int(w * 0.03)
    bar_y1 = int(h * 0.11)
    bar_x2 = int(w * 0.35)
    bar_y2 = int(h * 0.145)

    ratio = min(drowsy_counter / max(drowsy_threshold_frames, 1), 1.0)
    fill_x = int(bar_x1 + (bar_x2 - bar_x1) * ratio)

    cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 2)
    bar_color = (0, int(255 * (1 - ratio)), int(255 * ratio))
    cv2.rectangle(display, (bar_x1, bar_y1), (fill_x, bar_y2), bar_color, -1)
    cv2.putText(
        display,
        "Drowsiness level",
        (bar_x1, bar_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    right_x = int(w * 0.70)
    start_y = int(h * 0.06)
    line_h = int(max(22, h * 0.032))

    stats = [
        ("===== SYSTEM =====", (255, 255, 255)),
        (f"FPS: {fps:.2f}", (0, 255, 0)),
        ("", (255, 255, 255)),
        ("===== SCORES =====", (255, 255, 255)),
        (f"AVG EAR: {avg_ear:.3f}", (255, 255, 255)),
        (f"EAR Score: {ear_score:.3f}", (255, 255, 255)),
        (f"CNN Score: {cnn_score:.3f}", (255, 255, 255)),
        (f"Fused Score: {fused_score:.3f}", (0, 200, 255)),
        ("", (255, 255, 255)),
        ("===== COUNTERS =====", (255, 255, 255)),
        (f"Drowsy: {drowsy_counter}/{drowsy_threshold_frames}", (255, 255, 255)),
        (f"Recovery: {recovery_counter}/{recovery_threshold_frames}", (255, 255, 255)),
        (f"No face: {no_face_counter}", (255, 255, 255)),
    ]

    for i, (text, color) in enumerate(stats):
        y = start_y + i * line_h
        if y < h - 20:
            cv2.putText(
                display,
                text,
                (right_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    return display



def resolve_status_text(
    face_found: bool,
    alert_on: bool,
    fused_score: float,
    suspicious_threshold: float,
) -> tuple[str, tuple[int, int, int]]:
    if not face_found:
        return "NO FACE DETECTED", (0, 200, 255)
    if alert_on:
        return "DROWSY ALERT", (0, 0, 255)
    if fused_score >= suspicious_threshold:
        return "SUSPICIOUS", (0, 200, 255)
    return "AWAKE", (0, 255, 0)



def create_video_writer(output_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))



def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 10)
    return f"{m:02d}:{s:02d}.{ms}"



def draw_simple_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close(fig)



def draw_history_chart(df: pd.DataFrame, cols: list[str], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in cols:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# =========================
# Pages
# =========================
def render_page_eda() -> None:
    st.title("1. Giới thiệu & Khám phá dữ liệu (EDA)")
    st.subheader("Đề tài: Phát hiện trạng thái buồn ngủ của tài xế bằng mô hình MobileNetV2 kết hợp chỉ số khía cạnh mắt nhằm cảnh báo kịp thời")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Họ tên SV:** Phạm Văn Quân")
        st.markdown("**MSSV:** 22T1020347")
    with col2:
        st.markdown("**Mô tả bài toán:**")
        st.write(
            "Hệ thống hỗ trợ phát hiện dấu hiệu buồn ngủ của tài xế từ video thông qua "
            "Face Mesh, chỉ số EAR và mô hình MobileNetV2 phân loại trạng thái mắt. "
            "Mục tiêu là giảm nguy cơ tai nạn và tăng tính an toàn khi lái xe."
        )

    st.markdown("---")
    st.subheader("Dữ liệu thô")
    df = load_metadata_df()
    if df is None:
        st.warning(
            "Chưa có `data/sample_metadata.csv`. Hãy thêm file metadata để hiển thị EDA đầy đủ."
        )
        return

    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Biểu đồ phân tích dữ liệu")
    c1, c2 = st.columns(2)

    with c1:
        if "label" in df.columns:
            draw_simple_bar_chart(
                df["label"].value_counts(),
                "Phân phối nhãn",
                "Nhãn",
                "Số lượng mẫu",
            )
        else:
            st.info("Metadata chưa có cột `label`.")

    with c2:
        if "split" in df.columns:
            draw_simple_bar_chart(
                df["split"].value_counts(),
                "Phân chia dữ liệu train/val/test",
                "Tập dữ liệu",
                "Số lượng mẫu",
            )
        elif "width" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df["width"].dropna(), bins=20)
            ax.set_title("Phân bố chiều rộng ảnh")
            ax.set_xlabel("Width")
            ax.set_ylabel("Số lượng")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Metadata chưa có cột phù hợp cho biểu đồ thứ hai.")

    st.subheader("Nhận xét dữ liệu")
    notes = []
    if "label" in df.columns:
        vc = df["label"].value_counts(normalize=True)
        if len(vc) >= 2:
            imbalance = float(vc.max() - vc.min())
            if imbalance > 0.2:
                notes.append("Dữ liệu có dấu hiệu lệch nhãn, cần chú ý khi đánh giá mô hình.")
            else:
                notes.append("Phân phối nhãn tương đối cân bằng.")
    if {"width", "height"}.issubset(df.columns):
        notes.append("Bộ dữ liệu có thông tin kích thước ảnh, có thể dùng để kiểm tra tính đồng nhất khi tiền xử lý.")
    notes.append(
        "Các đặc trưng quan trọng nhất trong bài toán là trạng thái mắt, hình dạng mắt theo thời gian và chất lượng crop vùng mắt."
    )
    for note in notes:
        st.write(f"- {note}")



def render_page_deploy() -> None:
    st.title("2. Triển khai mô hình")
    st.caption("Upload video, hệ thống sẽ xử lý từng frame và hiển thị kết quả gần realtime.")

    st.sidebar.header("Cấu hình hệ thống")
    display_width = st.sidebar.selectbox("Kích thước xử lý frame", options=[480, 640, 800], index=1)
    frame_skip = st.sidebar.slider("Xử lý mỗi N frame", 1, 5, 1, 1)
    ui_update_every = st.sidebar.slider("Cập nhật giao diện mỗi N frame xử lý", 1, 10, 2, 1)
    alpha = st.sidebar.slider("Trọng số CNN trong fusion", 0.0, 1.0, 0.6, 0.05)
    ear_open = st.sidebar.slider("EAR mở mắt chuẩn", 0.20, 0.40, 0.30, 0.01)
    ear_closed = st.sidebar.slider("EAR nhắm mắt chuẩn", 0.10, 0.30, 0.18, 0.01)
    model_closed_threshold = st.sidebar.slider("Ngưỡng closed prob của model", 0.50, 0.95, 0.75, 0.01)
    suspicious_fused_threshold = st.sidebar.slider("Ngưỡng fused score nghi ngờ", 0.30, 0.95, 0.60, 0.01)
    alert_fused_threshold = st.sidebar.slider("Ngưỡng fused score cảnh báo", 0.40, 0.99, 0.70, 0.01)
    smoothing_window = st.sidebar.slider("Số frame làm mượt", 1, 30, 10, 1)
    drowsy_seconds_threshold = st.sidebar.slider("Số giây nghi ngủ gật để bật cảnh báo", 0.2, 3.0, 0.8, 0.1)
    recovery_seconds_threshold = st.sidebar.slider("Số giây hồi phục để tắt cảnh báo", 0.2, 3.0, 0.8, 0.1)
    draw_landmarks = st.sidebar.checkbox("Vẽ landmark mắt", value=True)
    draw_face_bbox = st.sidebar.checkbox("Vẽ khung mặt", value=False)
    draw_eye_boxes = st.sidebar.checkbox("Vẽ khung mắt", value=True)
    save_output_video = st.sidebar.checkbox("Lưu video kết quả", value=True)

    model_exists = MODEL_PATH.exists()
    label_exists = LABEL_MAP_PATH.exists()
    if not model_exists or not label_exists:
        st.warning(
            "Chưa tìm thấy model hoặc label map trong thư mục models/. "
            "Bạn cần tự thêm `eye_state_model_final.keras` và `label_map.json`."
        )

    uploaded_file = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is None:
        return

    st.subheader("Video gốc")
    st.video(uploaded_file)
    run_button = st.button("Bắt đầu xử lý", type="primary")

    if not run_button:
        return

    if not model_exists or not label_exists:
        st.error("Thiếu model hoặc label_map.json. Không thể chạy suy luận.")
        st.stop()

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.flush()
    input_video_path = temp_input.name

    try:
        classifier = load_classifier(str(MODEL_PATH), str(LABEL_MAP_PATH))
        detector = load_detector()
        alarm = AlarmPlayer(ALARM_PATH)
    except Exception as exc:
        st.error(f"Lỗi khởi tạo tài nguyên: {exc}")
        st.stop()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Không thể mở video đã upload.")
        st.stop()

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 1e-6:
        src_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"FPS nguồn: {src_fps:.2f} | Kích thước: {src_width}x{src_height} | Tổng frame: {total_frames}")

    effective_fps = src_fps / frame_skip
    drowsy_threshold_frames = max(1, int(drowsy_seconds_threshold * effective_fps))
    recovery_threshold_frames = max(1, int(recovery_seconds_threshold * effective_fps))

    frame_placeholder = st.empty()
    progress_placeholder = st.empty()
    stats_placeholder = st.empty()
    summary_placeholder = st.empty()
    audio_placeholder = st.empty()

    output_video_path = OUTPUT_DIR / f"processed_{int(time.time())}.mp4"
    writer = None

    score_buffer: deque[float] = deque(maxlen=smoothing_window)
    drowsy_counter = 0
    recovery_counter = 0
    no_face_counter = 0
    alarm_on = False

    frame_idx = 0
    processed_frames = 0
    suspicious_frames = 0
    alert_events_count = 0
    total_no_face_frames = 0

    prev_display_time = time.time()
    measured_fps = 0.0

    event_log: list[dict] = []
    alert_started_at_frame: int | None = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_skip > 1 and (frame_idx % frame_skip != 0):
                continue

            processed_frames += 1

            h0, w0 = frame.shape[:2]
            scale = display_width / float(w0)
            new_h = max(1, int(h0 * scale))
            frame = cv2.resize(frame, (display_width, new_h), interpolation=cv2.INTER_AREA)

            result = detector.process(
                frame,
                draw=True,
                draw_landmarks=draw_landmarks,
                draw_face_bbox=draw_face_bbox,
                draw_eye_boxes=draw_eye_boxes,
            )
            display = result.frame_bgr.copy()

            left_ear = right_ear = avg_ear = 0.0
            left_closed_prob = right_closed_prob = 0.0
            ear_score = cnn_score = fused_score = 0.0
            smooth_score = 0.0

            if not result.face_found:
                no_face_counter += 1
                total_no_face_frames += 1
                drowsy_counter = 0
                recovery_counter = 0

                if alarm_on:
                    alarm.stop()
                    alarm_on = False
                    if alert_started_at_frame is not None:
                        event_log.append({"time": format_seconds(frame_idx / src_fps), "event": "Alert ended (no face)"})
                        alert_started_at_frame = None
            else:
                no_face_counter = 0
                left_ear, right_ear, avg_ear = compute_average_ear(result.left_eye_points, result.right_eye_points)

                left_pred = classify_eye_state(classifier, result.left_eye_crop)
                right_pred = classify_eye_state(classifier, result.right_eye_crop)
                left_closed_prob = classifier.get_closed_prob(left_pred)
                right_closed_prob = classifier.get_closed_prob(right_pred)

                ear_score, cnn_score, fused_score = fuse_eye_scores(
                    avg_ear=avg_ear,
                    left_closed_prob=left_closed_prob,
                    right_closed_prob=right_closed_prob,
                    alpha=alpha,
                    ear_open=ear_open,
                    ear_closed=ear_closed,
                )

                score_buffer.append(fused_score)
                smooth_score = float(np.mean(score_buffer)) if score_buffer else fused_score

                model_closed = left_closed_prob >= model_closed_threshold or right_closed_prob >= model_closed_threshold
                suspicious_now = smooth_score >= suspicious_fused_threshold or model_closed
                alert_condition_now = smooth_score >= alert_fused_threshold or (avg_ear <= ear_closed and model_closed)

                if suspicious_now:
                    suspicious_frames += 1

                if alert_condition_now:
                    drowsy_counter += 1
                    recovery_counter = 0
                else:
                    drowsy_counter = max(0, drowsy_counter - 1)
                    recovery_counter += 1

                if drowsy_counter >= drowsy_threshold_frames and not alarm_on:
                    alarm.play()
                    alarm_on = True
                    alert_events_count += 1
                    alert_started_at_frame = frame_idx
                    event_log.append({"time": format_seconds(frame_idx / src_fps), "event": "Drowsy alert triggered"})

                if alarm_on and recovery_counter >= recovery_threshold_frames:
                    alarm.stop()
                    alarm_on = False
                    event_log.append({"time": format_seconds(frame_idx / src_fps), "event": "Recovered / alert cleared"})
                    alert_started_at_frame = None

            status_text, status_color = resolve_status_text(
                face_found=result.face_found,
                alert_on=alarm_on,
                fused_score=smooth_score,
                suspicious_threshold=suspicious_fused_threshold,
            )

            now = time.time()
            dt = max(now - prev_display_time, 1e-6)
            measured_fps = 1.0 / dt
            prev_display_time = now

            display = draw_status_ui(
                frame=display,
                status_text=status_text,
                status_color=status_color,
                fps=measured_fps,
                avg_ear=avg_ear,
                ear_score=ear_score,
                cnn_score=cnn_score,
                fused_score=smooth_score,
                drowsy_counter=drowsy_counter,
                recovery_counter=recovery_counter,
                no_face_counter=no_face_counter,
                drowsy_threshold_frames=drowsy_threshold_frames,
                recovery_threshold_frames=recovery_threshold_frames,
                alert_on=alarm_on,
            )

            render_browser_alarm(
                placeholder=audio_placeholder,
                alarm_wav_path=ALARM_PATH,
                should_play=alarm_on,
                key="drowsy_alarm_audio",
            )

            if save_output_video and writer is None:
                out_h, out_w = display.shape[:2]
                writer = create_video_writer(str(output_video_path), fps=src_fps if src_fps > 1 else 25.0, width=out_w, height=out_h)

            if writer is not None:
                writer.write(display)

            should_update_ui = processed_frames == 1 or (processed_frames % ui_update_every == 0)
            if should_update_ui:
                frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                progress_ratio = frame_idx / max(total_frames, 1)
                progress_placeholder.progress(min(int(progress_ratio * 100), 100), text=f"Đang xử lý frame {frame_idx}/{total_frames}")
                stats_placeholder.json(
                    {
                        "frame_index": frame_idx,
                        "processed_frames": processed_frames,
                        "face_found": result.face_found,
                        "avg_ear": round(float(avg_ear), 4),
                        "left_closed_prob": round(float(left_closed_prob), 4),
                        "right_closed_prob": round(float(right_closed_prob), 4),
                        "ear_score": round(float(ear_score), 4),
                        "cnn_score": round(float(cnn_score), 4),
                        "smooth_fused_score": round(float(smooth_score), 4),
                        "drowsy_counter": drowsy_counter,
                        "recovery_counter": recovery_counter,
                        "alarm_on": alarm_on,
                    }
                )

        cap.release()
        if writer is not None:
            writer.release()
        alarm.close()
        audio_placeholder.empty()

    except Exception as exc:
        cap.release()
        if writer is not None:
            writer.release()
        alarm.close()
        audio_placeholder.empty()
        st.error(f"Lỗi trong quá trình xử lý video: {exc}")
        st.stop()

    processed_duration_sec = processed_frames / max(effective_fps, 1e-6)
    suspicious_ratio = suspicious_frames / max(processed_frames, 1)

    summary_placeholder.success("Xử lý video hoàn tất.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số frame đã xử lý", processed_frames)
        st.metric("Số lần cảnh báo", alert_events_count)
    with col2:
        st.metric("Tỷ lệ frame nghi ngờ", f"{suspicious_ratio * 100:.2f}%")
        st.metric("No-face frames", total_no_face_frames)
    with col3:
        st.metric("Thời lượng xử lý tương đương", format_seconds(processed_duration_sec))
        st.metric("FPS xử lý cuối", f"{measured_fps:.2f}")

    st.subheader("Nhật ký sự kiện")
    if event_log:
        st.dataframe(event_log, use_container_width=True)
    else:
        st.info("Không có sự kiện cảnh báo nào được ghi nhận.")

    if save_output_video and output_video_path.exists():
        st.subheader("Video kết quả")
        with open(output_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button(
            label="Tải video đã xử lý",
            data=video_bytes,
            file_name=output_video_path.name,
            mime="video/mp4",
        )



def render_page_evaluation() -> None:
    st.title("3. Đánh giá & Hiệu năng (Evaluation)")
    st.write(
        "Trang này trình bày các chỉ số đánh giá của mô hình, biểu đồ kỹ thuật và phân tích sai số."
    )

    metrics = load_metrics_dict()
    cm_df = load_confusion_matrix_df()
    history_df = load_training_history_df()

    if metrics:
        st.subheader("Các chỉ số đánh giá")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        col3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        col4.metric("F1-score", f"{metrics.get('f1_score', 0):.4f}")
    else:
        st.warning("Chưa có `data/evaluation_metrics.json`.")

    st.subheader("Biểu đồ kỹ thuật")
    c1, c2 = st.columns(2)

    with c1:
        if cm_df is not None:
            st.write("**Confusion Matrix**")
            st.dataframe(cm_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm_df.values)
            ax.set_xticks(range(len(cm_df.columns)))
            ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(cm_df.index)))
            ax.set_yticklabels(cm_df.index)
            ax.set_title("Confusion Matrix")
            for i in range(cm_df.shape[0]):
                for j in range(cm_df.shape[1]):
                    ax.text(j, i, int(cm_df.values[i, j]), ha="center", va="center")
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Chưa có `data/confusion_matrix.csv`.")

    with c2:
        if history_df is not None and "epoch" in history_df.columns:
            if {"train_accuracy", "val_accuracy"}.intersection(history_df.columns):
                draw_history_chart(history_df, ["train_accuracy", "val_accuracy"], "Accuracy theo Epoch", "Accuracy")
            elif {"accuracy", "val_accuracy"}.issubset(history_df.columns):
                draw_history_chart(history_df, ["accuracy", "val_accuracy"], "Accuracy theo Epoch", "Accuracy")
            else:
                st.info("Training history chưa có cột accuracy phù hợp.")
        else:
            st.info("Chưa có `data/training_history.csv`.")

    if history_df is not None and "epoch" in history_df.columns:
        st.subheader("Đồ thị Loss")
        if {"train_loss", "val_loss"}.intersection(history_df.columns):
            draw_history_chart(history_df, ["train_loss", "val_loss"], "Loss theo Epoch", "Loss")
        elif {"loss", "val_loss"}.issubset(history_df.columns):
            draw_history_chart(history_df, ["loss", "val_loss"], "Loss theo Epoch", "Loss")

    st.subheader("Phân tích sai số và hướng cải thiện")
    st.write(
        "Mô hình thường dễ nhầm trong các trường hợp ánh sáng yếu, góc mặt lệch, mắt bị che bởi kính, "
        "video mờ hoặc crop vùng mắt chưa chuẩn. Phần EAR cũng có thể dao động khi Face Mesh không bám ổn định."
    )
    st.write(
        "Hướng cải thiện: tăng độ đa dạng dữ liệu, thêm đặc trưng theo thời gian như PERCLOS hoặc head pose, "
        "tối ưu ngưỡng cho từng người, và giảm độ trễ bằng cách chỉ cập nhật giao diện theo chu kỳ."
    )


# =========================
# App router
# =========================
st.sidebar.title("Điều hướng")
page = st.sidebar.radio(
    "Chọn trang",
    [
        "1. Giới thiệu & EDA",
        "2. Triển khai mô hình",
        "3. Đánh giá & Hiệu năng",
    ],
)

if page == "1. Giới thiệu & EDA":
    render_page_eda()
elif page == "2. Triển khai mô hình":
    render_page_deploy()
else:
    render_page_evaluation()


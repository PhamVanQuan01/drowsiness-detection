from __future__ import annotations

from pathlib import Path
from collections import deque
import tempfile
import time
import base64
import json
import subprocess
import shutil

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio_ffmpeg
from src.alarm import AlarmPlayer
from src.ear import compute_average_ear, ear_to_closed_score
from src.face_mesh_utils import FaceMeshDetector
from src.inference import EyeStateClassifier


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "eye_state_model_final.h5"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map.json"
ALARM_PATH = BASE_DIR / "assets" / "alarm.wav"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_METADATA_PATH    = DATA_DIR / "sample_metadata.csv"
EVALUATION_METRICS_PATH = DATA_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH   = DATA_DIR / "confusion_matrix.csv"
TRAINING_HISTORY_PATH   = DATA_DIR / "training_history.csv"


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
        return {"label": "Unknown", "confidence": 0.0, "probs": {}}
    return classifier.predict(eye_crop)


def fuse_eye_scores(
    avg_ear: float,
    left_closed_prob: float,
    right_closed_prob: float,
    alpha: float = 0.6,
    ear_open: float = 0.30,
    ear_closed: float = 0.18,
) -> tuple[float, float, float]:
    ear_score   = ear_to_closed_score(ear=avg_ear, ear_open=ear_open, ear_closed=ear_closed)
    cnn_score   = (left_closed_prob + right_closed_prob) / 2.0
    fused_score = alpha * cnn_score + (1.0 - alpha) * ear_score
    return float(ear_score), float(cnn_score), float(fused_score)


def draw_header_panel(frame: np.ndarray, panel_height_ratio: float = 0.18) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = int(h * panel_height_ratio)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)


def draw_status_ui(
    frame, status_text, status_color, fps, avg_ear,
    ear_score, cnn_score, fused_score,
    drowsy_counter, recovery_counter, no_face_counter,
    drowsy_threshold_frames, recovery_threshold_frames, alert_on,
) -> np.ndarray:
    display    = draw_header_panel(frame)
    h, w       = display.shape[:2]
    font_scale = max(w / 1100.0, 0.55)
    thickness  = max(int(font_scale * 2), 1)

    cv2.putText(display, status_text, (int(w * 0.03), int(h * 0.07)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness + 1, cv2.LINE_AA)

    if alert_on:
        cv2.rectangle(display, (10, 10), (w - 10, h - 10), (0, 0, 255), 3)
        cv2.putText(display, "DROWSY ALERT!", (int(w * 0.03), int(h * 0.15)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.95,
                    (0, 0, 255), thickness + 1, cv2.LINE_AA)

    bar_x1, bar_y1 = int(w * 0.03), int(h * 0.11)
    bar_x2, bar_y2 = int(w * 0.35), int(h * 0.145)

    # UI Bar cải tiến: hiển thị Recovery nếu đang alarm, ngược lại hiển thị Drowsy
    if alert_on:
        current_val = recovery_counter
        current_thr = max(recovery_threshold_frames, 1)
        bar_label   = "Recovery progress"
        bar_color   = (0, 255, 0) # Xanh lá cho hồi phục
    else:
        current_val = drowsy_counter
        current_thr = max(drowsy_threshold_frames, 1)
        bar_label   = "Drowsiness level"
        ratio       = min(current_val / current_thr, 1.0)
        bar_color   = (0, int(255 * (1 - ratio)), int(255 * ratio))

    ratio  = min(current_val / current_thr, 1.0)
    fill_x = int(bar_x1 + (bar_x2 - bar_x1) * ratio)
    cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 2)
    cv2.rectangle(display, (bar_x1, bar_y1), (fill_x, bar_y2), bar_color, -1)
    cv2.putText(display, bar_label, (bar_x1, bar_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    right_x = int(w * 0.70)
    start_y = int(h * 0.06)
    line_h  = int(max(22, h * 0.032))
    stats = [
        ("===== SYSTEM =====",                               (255, 255, 255)),
        (f"FPS: {fps:.2f}",                                  (0, 255, 0)),
        ("",                                                  (255, 255, 255)),
        ("===== SCORES =====",                               (255, 255, 255)),
        (f"AVG EAR: {avg_ear:.3f}",                          (255, 255, 255)),
        (f"EAR Score: {ear_score:.3f}",                      (255, 255, 255)),
        (f"CNN Score: {cnn_score:.3f}",                      (255, 255, 255)),
        (f"Fused Score: {fused_score:.3f}",                  (0, 200, 255)),
        ("",                                                  (255, 255, 255)),
        ("===== COUNTERS =====",                             (255, 255, 255)),
        (f"Drowsy: {drowsy_counter}/{drowsy_threshold_frames}",       (255, 255, 255)),
        (f"Recovery: {recovery_counter}/{recovery_threshold_frames}", (255, 255, 255)),
        (f"No face: {no_face_counter}",                      (255, 255, 255)),
    ]
    for i, (text, color) in enumerate(stats):
        y = start_y + i * line_h
        if y < h - 20:
            cv2.putText(display, text, (right_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 0.42, color, 1, cv2.LINE_AA)
    return display


def resolve_status_text(face_found, alert_on, fused_score, suspicious_threshold):
    if not face_found:
        return "NO FACE DETECTED", (0, 200, 255)
    if alert_on:
        return "DROWSY ALERT", (0, 0, 255)
    if fused_score >= suspicious_threshold:
        return "SUSPICIOUS", (0, 200, 255)
    return "AWAKE", (0, 255, 0)


def get_ffmpeg_exe() -> str | None:
    """Ưu tiên ffmpeg system (có libx264), fallback imageio_ffmpeg."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def create_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Thử ghi H.264 (avc1) trực tiếp — trình duyệt phát được, thời lượng đúng.
    Fallback sang mp4v nếu avc1 không khả dụng.
    """
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    # last resort
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def reencode_video_h264(input_path: Path, output_path: Path, fps: float) -> bool:
    """
    Re-encode sang H.264 bằng ffmpeg system hoặc imageio_ffmpeg.
    Ép -r fps để thời lượng output = thời lượng input.
    Fallback: nếu libx264 không có thì copy stream + fix container.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    if not ffmpeg_exe:
        return False
    try:
        # Thử encode libx264
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(input_path),
            "-r", str(fps),
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True

        # libx264 không có → chỉ copy stream, fix container & fps metadata
        if output_path.exists():
            output_path.unlink()
        cmd2 = [
            ffmpeg_exe, "-y",
            "-i", str(input_path),
            "-r", str(fps),
            "-vcodec", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True

        st.warning(f"FFmpeg lỗi:\n{result.stderr[-800:]}")
        return False
    except Exception as e:
        st.warning(f"Exception re-encode: {e}")
        return False


def mix_alarm_into_video(
    video_path: Path,
    alarm_path: Path,
    output_path: Path,
    alarm_intervals: list[tuple[float, float]],
    video_duration: float,
) -> bool:
    """Mix alarm audio segments into the video while preserving full video duration."""
    if not alarm_intervals or not alarm_path.exists():
        return False

    ffmpeg_exe = get_ffmpeg_exe()
    if not ffmpeg_exe:
        return False

    try:
        filter_parts = []
        # Nền im lặng được CẮT ĐÚNG bằng thời lượng video để tránh treo
        filter_parts.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{video_duration:.3f}[a_base]")

        # Tạo từng đoạn alarm theo interval
        for idx, (start, end) in enumerate(alarm_intervals):
            duration = max(end - start, 0.1)
            delay_ms = int(start * 1000)
            filter_parts.append(
                f"[1:a]atrim=0:{duration:.3f},"
                f"aloop=loop=-1:size=999999," # Lặp tiếng chuông nếu interval dài hơn file âm thanh
                f"atrim=0:{duration:.3f},"
                f"adelay={delay_ms}|{delay_ms}[a{idx}]"
            )

        mix_inputs = "[a_base]" + "".join(f"[a{i}]" for i in range(len(alarm_intervals)))
        # Mix nền + các đoạn alarm, lấy duration=first (theo nền a_base đã trim)
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(alarm_intervals)+1}:duration=first[aout]"
        )

        filter_complex = ";".join(filter_parts)

        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-i", str(alarm_path),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path),
        ]

        # Thêm timeout 120s để không treo Streamlit mãi mãi
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True

        st.warning(f"Mix audio lỗi:\n{result.stderr[-800:]}")
        return False
    except subprocess.TimeoutExpired:
        st.warning("Mix audio bị timeout (quá 120s). Video sẽ được xuất không kèm tiếng cảnh báo.")
        return False
    except Exception as e:
        st.warning(f"Exception mix audio: {e}")
        return False


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m  = int(seconds // 60)
    s  = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 10)
    return f"{m:02d}:{s:02d}.{ms}"


def draw_simple_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.xticks(rotation=0)
    st.pyplot(fig); plt.close(fig)


def draw_history_chart(df: pd.DataFrame, cols: list[str], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in cols:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.legend()
    st.pyplot(fig); plt.close(fig)


# =========================
# Chart helpers
# =========================

def plot_label_distribution():
    labels = ["Closed_Eyes", "Open_Eyes"]
    counts = [2726, 2726]
    colors = ["#e74c3c", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
    ax.set_title("Phân phối nhãn (Label Distribution)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Số lượng mẫu", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{count:,}\n({count / sum(counts) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)


def plot_split_distribution():
    splits = ["Train (77.6%)", "Val (12.3%)", "Test (10.1%)"]
    sizes  = [4233, 672, 547]
    colors = ["#3498db", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=splits, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold")
    autotexts[2].set_text("10.1%")
    ax.set_title("Phân chia dữ liệu Train / Val / Test\n(Tổng: 5,452 ảnh)",
                 fontsize=12, fontweight="bold", pad=12)
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre_circle)
    ax.text(0, 0, "5,452\nảnh", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#555")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)


def plot_split_label_breakdown():
    splits = ["Train", "Val", "Test"]
    closed = [2029, 336, 361]
    open_e = [2204, 336, 186]
    x = np.arange(len(splits)); width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width / 2, closed, width, label="Closed_Eyes",
                color="#e74c3c", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width / 2, open_e, width, label="Open_Eyes",
                color="#2ecc71", alpha=0.85, edgecolor="white")
    ax.set_title("Phân phối nhãn theo từng tập dữ liệu", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylabel("Số lượng mẫu", fontsize=10); ax.legend(fontsize=10)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)


def plot_ear_distribution():
    """
    Phân phối EAR khớp với tham số mặc định trong code:
      ear_open   = 0.30   (ngưỡng mở mắt chuẩn)
      ear_closed = 0.18   (ngưỡng nhắm mắt)
    → Closed_Eyes tập trung quanh μ≈0.14 (dưới ngưỡng 0.18)
    → Open_Eyes   tập trung quanh μ≈0.30 (vùng mở chuẩn)
    """
    x = np.linspace(0, 0.50, 500)
    # Closed: μ=0.14, σ=0.025  (rơi vào vùng < ear_closed=0.18)
    closed_dist = np.exp(-0.5 * ((x - 0.14) / 0.025) ** 2)
    # Open:   μ=0.30, σ=0.040  (rơi vào vùng ≈ ear_open=0.30)
    open_dist   = np.exp(-0.5 * ((x - 0.30) / 0.040) ** 2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, closed_dist, alpha=0.45, color="#e74c3c", label="Closed_Eyes (μ≈0.14)")
    ax.fill_between(x, open_dist,   alpha=0.45, color="#2ecc71", label="Open_Eyes  (μ≈0.30)")
    ax.plot(x, closed_dist, color="#c0392b", linewidth=1.8)
    ax.plot(x, open_dist,   color="#27ae60", linewidth=1.8)

    # Ngưỡng khớp với code: ear_closed_thresh=0.18, ear_open=0.30
    ax.axvline(0.18, color="#f39c12", linewidth=2,   linestyle="--",
               label="Ngưỡng EAR nhắm = 0.18  (ear_closed)")
    ax.axvline(0.30, color="#3498db", linewidth=1.5, linestyle=":",
               label="Ngưỡng EAR mở  = 0.30  (ear_open)")

    ax.set_title("Phân phối EAR theo nhãn (ước tính)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Giá trị EAR", fontsize=11)
    ax.set_ylabel("Mật độ xác suất (chuẩn hóa)", fontsize=10)
    ax.legend(fontsize=9); ax.set_xlim(0, 0.50)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)


def plot_ear_diagram():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
    eye_x     = np.linspace(1.5, 7.5, 200)
    eye_y_top = 2.0 + 0.9 * np.sin(np.pi * (eye_x - 1.5) / 6.0)
    eye_y_bot = 2.0 - 0.9 * np.sin(np.pi * (eye_x - 1.5) / 6.0)
    ax.fill_between(eye_x, eye_y_bot, eye_y_top, color="#2c3e50", alpha=0.5)
    ax.plot(eye_x, eye_y_top, color="#00bcd4", linewidth=1.5)
    ax.plot(eye_x, eye_y_bot, color="#00bcd4", linewidth=1.5)
    pts = {"p1": (1.5, 2.0), "p2": (3.0, 2.75), "p3": (5.5, 2.75),
           "p4": (7.5, 2.0), "p5": (5.5, 1.25), "p6": (3.0, 1.25)}
    corner_color = "#f39c12"; lid_color = "#00e5ff"
    for name, (px, py) in pts.items():
        color = corner_color if name in ("p1", "p4") else lid_color
        ax.plot(px, py, "o", color=color, markersize=9, zorder=5)
        offset = {"p1": (-0.35, 0), "p2": (0, 0.25), "p3": (0, 0.25),
                  "p4": (0.25, 0), "p5": (0, -0.32), "p6": (0, -0.32)}
        dx, dy = offset[name]
        ax.text(px + dx, py + dy, name, color="white", fontsize=9, fontweight="bold", ha="center")
    ax.annotate("", xy=(3.0, 1.25), xytext=(3.0, 2.75),
                arrowprops=dict(arrowstyle="<->", color="#2ecc71", lw=1.5))
    ax.text(2.55, 2.0, "||p2-p6||", color="#2ecc71", fontsize=8, ha="center", rotation=90)
    ax.annotate("", xy=(5.5, 1.25), xytext=(5.5, 2.75),
                arrowprops=dict(arrowstyle="<->", color="#2ecc71", lw=1.5))
    ax.text(5.95, 2.0, "||p3-p5||", color="#2ecc71", fontsize=8, ha="center", rotation=90)
    ax.annotate("", xy=(7.5, 1.6), xytext=(1.5, 1.6),
                arrowprops=dict(arrowstyle="<->", color=corner_color, lw=1.5))
    ax.text(4.5, 1.35, "||p1-p4||", color=corner_color, fontsize=8, ha="center")
    ax.text(4.5, 0.45, "EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)",
            color="#ecf0f1", fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1b2a",
                      edgecolor="#00bcd4", linewidth=1.5))
    ax.set_title("Sơ đồ 6 điểm landmark tính EAR", color="white",
                 fontsize=11, fontweight="bold", pad=8)
    st.pyplot(fig); plt.close(fig)


def plot_mobilenet_architecture():
    """
    Pipeline khớp với LaTeX:
    Input 224×224×3
    → Normalize [-1,1]
    → Data Augmentation (train only)
    → MobileNetV2 pretrained ImageNet (include_top=False)
    → GlobalAveragePooling2D
    → Dropout(0.3)
    → Dense(1, sigmoid)
    → Xác suất Open/Closed
    """
    blocks = [
        ("Input\n224×224×3",              "#8e44ad"),
        ("Normalize\n[-1, 1]",            "#2980b9"),
        ("Data Aug\n(train only)",         "#2980b9"),
        ("MobileNetV2\n(ImageNet)\ninclude_top=False", "#2980b9"),
        ("GlobalAvg\nPool2D",             "#16a085"),
        ("Dropout(0.3)",                  "#16a085"),
        ("Dense(1)\nSigmoid",             "#16a085"),
        ("Xác suất\nOpen / Closed",        "#27ae60"),
    ]
    fig, ax = plt.subplots(figsize=(14, 2.6))
    ax.axis("off"); ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
    n = len(blocks)
    for i, (label, color) in enumerate(blocks):
        x = i / (n - 1)
        rect = mpatches.FancyBboxPatch(
            (x - 0.055, 0.10), 0.11, 0.80,
            boxstyle="round,pad=0.02",
            facecolor=color + "33", edgecolor=color, linewidth=1.8,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(x, 0.50, label, transform=ax.transAxes, color="white",
                fontsize=7.5, ha="center", va="center", multialignment="center")
        if i < n - 1:
            ax.annotate(
                "", xy=((i + 1) / (n - 1) - 0.057, 0.50),
                xytext=(x + 0.057, 0.50),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#95a5a6", lw=1.2),
            )
    ax.set_title(
        "Kiến trúc Pipeline MobileNetV2 (Transfer Learning từ ImageNet)",
        color="white", fontsize=11, fontweight="bold", pad=10,
    )
    st.pyplot(fig); plt.close(fig)


def plot_fusion_diagram():
    """
    Sơ đồ fusion khớp với hàm fuse_eye_scores() trong code:
      ear_score  = ear_to_closed_score(avg_ear, ear_open=0.30, ear_closed=0.18)
      cnn_score  = (left_closed_prob + right_closed_prob) / 2.0
      fused_score = alpha * cnn_score + (1 - alpha) * ear_score
    Sau đó qua deque smoothing → quyết định AWAKE / SUSPICIOUS / ALERT.
    """
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off"); ax.set_facecolor("#0f1923"); fig.patch.set_facecolor("#0f1923")

    def box(cx, cy, text, color, width=0.14, height=0.32):
        rect = mpatches.FancyBboxPatch(
            (cx - width / 2, cy - height / 2), width, height,
            boxstyle="round,pad=0.015", facecolor=color + "22", edgecolor=color,
            linewidth=1.8, transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, transform=ax.transAxes, color="white",
                fontsize=8, ha="center", va="center", multialignment="center")

    def arrow(x1, y1, x2, y2, label="", color="#95a5a6"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.06, label, transform=ax.transAxes,
                    color=color, fontsize=7, ha="center")

    # Nodes
    box(0.05, 0.75, "Left Eye\nCrop",                    "#9b59b6")
    box(0.05, 0.25, "Right Eye\nCrop",                   "#9b59b6")
    box(0.20, 0.50, "MediaPipe\nFace Mesh",              "#2980b9")
    box(0.35, 0.75, "CNN Output\nP(closed)_L",           "#e67e22")
    box(0.35, 0.25, "CNN Output\nP(closed)_R",           "#e67e22")
    # --- SỬA: label EAR khớp với ear_to_closed_score() ---
    box(0.35, 0.50, "EAR Score\n→ [0,1]\n(closed score)", "#16a085")
    box(0.52, 0.65, "CNN Score\n= avg(L, R)",            "#e67e22")
    # --- SỬA: công thức rõ ràng hơn ---
    box(0.68, 0.50, "Fused Score\n= α·CNN\n+(1-α)·EAR", "#c0392b")
    box(0.83, 0.50, "Temporal\nSmoothing\n(Moving Avg\nN frames)", "#3498db")
    box(0.96, 0.75, "AWAKE\n✅",      "#27ae60")
    box(0.96, 0.50, "SUSPICIOUS\n⚠️", "#f39c12")
    box(0.96, 0.25, "ALERT\n🚨",      "#e74c3c")

    # Arrows
    arrow(0.11, 0.75, 0.16, 0.62)
    arrow(0.11, 0.25, 0.16, 0.38)
    arrow(0.24, 0.62, 0.30, 0.55)
    arrow(0.24, 0.68, 0.30, 0.75)
    arrow(0.24, 0.38, 0.30, 0.25)
    arrow(0.40, 0.75, 0.46, 0.70)
    arrow(0.40, 0.25, 0.46, 0.60)
    arrow(0.40, 0.50, 0.62, 0.52)          # EAR → Fused (trực tiếp)
    arrow(0.58, 0.65, 0.62, 0.58)          # CNN Score → Fused
    arrow(0.74, 0.50, 0.77, 0.50)          # Fused → Smoothing
    arrow(0.89, 0.58, 0.91, 0.73, "< suspicious_thresh",  "#27ae60")
    arrow(0.89, 0.50, 0.91, 0.50, "≥ suspicious_thresh",  "#f39c12")
    arrow(0.89, 0.42, 0.91, 0.27, "≥ alert_thresh + time","#e74c3c")

    ax.set_title(
        "Sơ đồ luồng Fusion Score  (EAR Score + CNN Score → Smoothing → Quyết định)",
        color="white", fontsize=11, fontweight="bold", pad=8,
    )
    st.pyplot(fig); plt.close(fig)


def plot_state_machine():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_facecolor("#0f1923"); fig.patch.set_facecolor("#0f1923")
    for cx, cy, label, color, r in [
        (1.8, 2.0, "AWAKE\n😊",        "#27ae60", 0.85),
        (5.0, 2.0, "SUSPICIOUS\n😐",   "#f39c12", 0.85),
        (8.2, 2.0, "DROWSY ALERT\n🚨", "#e74c3c", 0.95),
    ]:
        ax.add_patch(plt.Circle((cx, cy), r, facecolor=color + "22",
                                edgecolor=color, linewidth=2.2))
        ax.text(cx, cy, label, color="white", fontsize=9.5,
                ha="center", va="center", fontweight="bold", multialignment="center")
    ax.annotate("", xy=(4.12, 2.25), xytext=(2.65, 2.25),
                arrowprops=dict(arrowstyle="-|>", color="#f39c12", lw=1.8))
    ax.text(3.38, 2.55, "smooth_score\n≥ suspicious", color="#f39c12", fontsize=7.5, ha="center")
    ax.annotate("", xy=(7.27, 2.25), xytext=(5.88, 2.25),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1.8))
    ax.text(6.58, 2.55, "alert_condition_now\nduy trì N frame", color="#e74c3c", fontsize=7.5, ha="center")
    ax.annotate("", xy=(2.65, 1.75), xytext=(4.12, 1.75),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5))
    ax.text(3.38, 1.45, "smooth_score\n< suspicious", color="#27ae60", fontsize=7.5, ha="center")
    ax.annotate("", xy=(5.88, 1.75), xytext=(7.27, 1.75),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5))
    ax.text(6.58, 1.45, "recovery_counter\nđủ M frame", color="#27ae60", fontsize=7.5, ha="center")
    ax.annotate("", xy=(1.8, 1.0), xytext=(5.0, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#95a5a6", lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(3.38, 0.45, "No face → reset counters, stop alarm",
            color="#95a5a6", fontsize=7.5, ha="center")
    ax.set_title("Máy trạng thái cảnh báo buồn ngủ", color="white",
                 fontsize=11, fontweight="bold", pad=8)
    st.pyplot(fig); plt.close(fig)


def plot_confusion_matrix_custom():
    cm = np.array([[349, 12], [0, 186]]); labels = ["Closed_Eyes", "Open_Eyes"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Test Set (547 mẫu)", fontsize=12, fontweight="bold", pad=12)
    cell_labels = [["TP = 349\n(63.8%)", "FN = 12\n(2.2%)"],
                   ["FP = 0\n(0.0%)",    "TN = 186\n(34.0%)"]]
    colors_text = [["white", "#1a3a6e"], ["#1a3a6e", "white"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=colors_text[i][j])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_training_curves():
    epochs     = list(range(1, 16))
    train_loss = [0.1123, 0.0348, 0.0203, 0.0178, 0.0148, 0.0103, 0.0107, 0.0070, 0.0086, 0.0067, 0.0792, 0.0245, 0.0185, 0.0106, 0.0098]
    val_loss   = [0.0803, 0.0623, 0.0574, 0.0564, 0.0526, 0.0537, 0.0485, 0.0607, 0.0450, 0.0305, 0.0575, 0.0540, 0.0643, 0.0767, 0.0772]
    train_acc  = [0.9672, 0.9913, 0.9948, 0.9947, 0.9946, 0.9963, 0.9968, 0.9980, 0.9965, 0.9981, 0.9712, 0.9917, 0.9922, 0.9964, 0.9972]
    val_acc    = [0.9732, 0.9777, 0.9777, 0.9762, 0.9777, 0.9777, 0.9792, 0.9777, 0.9807, 0.9821, 0.9792, 0.9792, 0.9792, 0.9762, 0.9762]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, train_loss, "o-", color="#3498db", linewidth=2, markersize=5, label="Train Loss")
    ax1.plot(epochs, val_loss,   "s-", color="#e67e22", linewidth=2, markersize=5, label="Val Loss")
    ax1.axvline(11, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.text(11.2, max(train_loss) * 0.9, "LR reset\n(fine-tune)", color="#e74c3c", fontsize=8)
    ax1.set_title("Loss theo Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3); ax1.set_facecolor("#f8f9fa")
    ax2.plot(epochs, train_acc, "o-", color="#27ae60", linewidth=2, markersize=5, label="Train Accuracy")
    ax2.plot(epochs, val_acc,   "s-", color="#8e44ad", linewidth=2, markersize=5, label="Val Accuracy")
    best_epoch = val_acc.index(max(val_acc)) + 1
    ax2.axvline(best_epoch, color="#f39c12", linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.text(best_epoch + 0.2, min(val_acc) + 0.001,
             f"Best epoch {best_epoch}\nVal={max(val_acc):.4f}", color="#f39c12", fontsize=8)
    ax2.set_title("Accuracy theo Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3); ax2.set_facecolor("#f8f9fa")
    plt.suptitle("Quá trình huấn luyện MobileNetV2 — 15 Epochs",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_metrics_bar():
    metrics_data = {
        "Precision": {"Closed_Eyes": 1.0000, "Open_Eyes": 0.9394, "Macro avg": 0.9697},
        "Recall":    {"Closed_Eyes": 0.9668, "Open_Eyes": 1.0000, "Macro avg": 0.9834},
        "F1-Score":  {"Closed_Eyes": 0.9831, "Open_Eyes": 0.9688, "Macro avg": 0.9759},
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3); width = 0.22
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    classes = ["Closed_Eyes", "Open_Eyes", "Macro avg"]
    for idx, (metric, vals) in enumerate(metrics_data.items()):
        ys   = [vals[c] for c in classes]
        bars = ax.bar(x + idx * width, ys, width, label=metric,
                      color=colors[idx], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)
    ax.set_xticks(x + width); ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0.85, 1.04); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("So sánh Precision / Recall / F1 theo lớp",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


# =========================
# Pages
# =========================

def render_page_eda() -> None:
    st.title("🚀 1. Giới thiệu & Khám phá dữ liệu")
    st.markdown("---")
    st.markdown(
        "<h3 style='text-align:center;color:#1f77b4;'>"
        "ĐỀ TÀI: PHÁT HIỆN TRẠNG THÁI BUỒN NGỦ CỦA TÀI XẾ BẰNG MÔ HÌNH "
        "MOBILENETV2 KẾT HỢP CHỈ SỐ EAR NHẰM CẢNH BÁO KỊP THỜI"
        "</h3>",
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("**Thông tin sinh viên**\n\n🎓 **Họ và tên:** Phạm Văn Quân\n\n🆔 **Mã sinh viên:** 22T1020347")
    with col2:
        st.success(
            "**Giới thiệu**\n\n"
            "Tai nạn giao thông do buồn ngủ là một trong những nguyên nhân hàng đầu gây ra thiệt hại nghiêm trọng. "
            "Bài toán này nhằm **theo dõi và phân tích chỉ số khía cạnh mắt (EAR)** của tài xế qua camera "
            "và sẽ lập tức phát tín hiệu cảnh báo khi tài xế ngủ gật."
        )

    st.markdown("---")
    st.markdown("### 🧠 Lý thuyết Mô Hình & Phương Pháp Nhận Diện")

    with st.expander("📌 1. Trích xuất đặc trưng hình học với MediaPipe & EAR", expanded=True):
        c_left, c_right = st.columns([1.1, 1])
        with c_left:
            st.markdown("""
**MediaPipe Face Mesh** cung cấp **468 điểm mốc 3D** trên khuôn mặt.

$$EAR = \\frac{\\|p_2 - p_6\\| + \\|p_3 - p_5\\|}{2 \\times \\|p_1 - p_4\\|}$$

| Trạng thái | Giá trị EAR |
|---|---|
| Mắt mở | 0.25 – 0.35 |
| Nghi ngờ | 0.18 – 0.25 |
| Nhắm (buồn ngủ) | < 0.18 |
""")
        with c_right:
            plot_ear_diagram()

    with st.expander("📌 2. Phân loại trạng thái mắt bằng MobileNetV2", expanded=True):
        st.markdown("""
**MobileNetV2** pretrained từ ImageNet → GlobalAveragePooling2D → Dropout → Dense(1, sigmoid).

- **Inverted Residual Blocks**: Giảm ~8–9× tham số so với VGG16.
- **Linear Bottlenecks**: Tránh mất mát thông tin ở lớp cuối block.
""")
        plot_mobilenet_architecture()

    with st.expander("📌 3. Fusion Score (EAR + CNN)", expanded=True):
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("""
1. **EAR → Score** [0,1]
2. **CNN → Score**: avg(P_closed_L, P_closed_R)
3. **Fusion**: α·CNN + (1-α)·EAR
4. **Smoothing**: Moving Average N frame
5. **Counter**: Cảnh báo khi duy trì đủ frame
""")
        with c_right:
            plot_fusion_diagram()

    with st.expander("📌 4. Cơ chế Counter-based Alert", expanded=False):
        st.markdown("""
**Bật alarm:** Fused score cao liên tục đủ giây → bật. Nháy mắt xen kẽ → reset ngay.

**Tắt alarm:** Chỉ dùng EAR. EAR > ngưỡng liên tục đủ giây → tắt. Nhắm lại → reset ngay.
```python
# Bật alarm
if alert_condition_now:  drowsy_counter += 1
else:                    drowsy_counter = 0   # reset ngay

# Tắt alarm (chỉ EAR)
if avg_ear > ear_closed_thresh:  recovery_counter += 1
else:                            recovery_counter = 0  # reset ngay
```
""")
        plot_state_machine()

    st.markdown("---")
    st.markdown("### 📁 Cấu trúc Dataset")
    st.code(
        "dataset/\n├── train/\n│   ├── Closed_Eyes/ (2029)\n│   └── Open_Eyes/   (2204)\n"
        "├── val/\n│   ├── Closed_Eyes/ (336)\n│   └── Open_Eyes/   (336)\n"
        "└── test/\n    ├── Closed_Eyes/ (361)\n    └── Open_Eyes/   (186)",
        language="text",
    )

    st.markdown("---")
    st.subheader("📊 Biểu đồ phân bố dữ liệu")
    c1, c2 = st.columns(2)
    with c1:
        df_meta_loaded = load_metadata_df()
        if df_meta_loaded is not None and "label" in df_meta_loaded.columns:
            draw_simple_bar_chart(df_meta_loaded["label"].value_counts(),
                                  "Phân phối nhãn", "Nhãn", "Số lượng mẫu")
        else:
            plot_label_distribution()
    with c2:
        if df_meta_loaded is not None and "split" in df_meta_loaded.columns:
            draw_simple_bar_chart(df_meta_loaded["split"].value_counts(),
                                  "Train/Val/Test", "Tập dữ liệu", "Số lượng")
        else:
            plot_split_distribution()

    c3, c4 = st.columns(2)
    with c3:
        plot_split_label_breakdown()
    with c4:
        plot_ear_distribution()

    st.markdown("---")
    st.markdown("### 💡 Nhận xét")
    st.success("✅ Dữ liệu cân bằng nhãn ~50–50.")
    st.success("✅ Kích thước đồng nhất 224×224 px.")
    st.info("📌 EAR có vùng overlap → cần kết hợp CNN → Fusion Score.")


# =========================
# Trang 2: Triển khai mô hình
# =========================

def render_page_deploy() -> None:
    st.title("2. Triển khai mô hình")
    st.caption("Upload video, hệ thống xử lý và trả về video kết quả cùng bảng phân tích.")

    st.sidebar.header("Cấu hình hệ thống")
    display_width              = st.sidebar.selectbox("Kích thước xử lý frame", [480, 640, 800], index=1)
    frame_skip                 = st.sidebar.slider("Xử lý mỗi N frame", 1, 5, 1, 1)
    alpha                      = st.sidebar.slider("Trọng số CNN trong fusion", 0.0, 1.0, 0.35, 0.05)
    ear_open                   = st.sidebar.slider("EAR mở mắt chuẩn", 0.20, 0.40, 0.30, 0.01)
    ear_closed_thresh          = st.sidebar.slider("EAR nhắm mắt chuẩn", 0.10, 0.30, 0.18, 0.01)
    model_closed_threshold     = st.sidebar.slider("Ngưỡng closed prob của model", 0.50, 0.95, 0.75, 0.01)
    suspicious_fused_threshold = st.sidebar.slider("Ngưỡng fused score nghi ngờ", 0.30, 0.95, 0.60, 0.01)
    alert_fused_threshold      = st.sidebar.slider("Ngưỡng fused score cảnh báo", 0.40, 0.99, 0.75, 0.01)
    smoothing_window           = st.sidebar.slider("Số frame làm mượt", 1, 30, 5, 1)
    drowsy_seconds_threshold   = st.sidebar.slider("Nhắm mắt liên tục để bật cảnh báo (giây)", 0.5, 5.0, 2.0, 0.1)
    recovery_seconds_threshold = st.sidebar.slider("Mở mắt liên tục để tắt cảnh báo (giây)", 0.5, 5.0, 1.5, 0.1)
    draw_landmarks             = st.sidebar.checkbox("Vẽ landmark mắt", value=True)
    draw_face_bbox             = st.sidebar.checkbox("Vẽ khung mặt", value=False)
    draw_eye_boxes             = st.sidebar.checkbox("Vẽ khung mắt", value=True)
    mix_alarm_audio            = st.sidebar.checkbox("Mix tiếng cảnh báo vào video", value=True)

    model_exists = MODEL_PATH.exists()
    label_exists = LABEL_MAP_PATH.exists()
    if not model_exists or not label_exists:
        st.warning("Chưa tìm thấy model hoặc label map trong thư mục models/.")

    uploaded_file = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is None:
        return

    st.subheader("🎬 Video gốc")
    st.video(uploaded_file)
    run_button = st.button("▶️ Bắt đầu xử lý", type="primary")
    if not run_button:
        return

    if not model_exists or not label_exists:
        st.error("Thiếu model hoặc label_map.json.")
        st.stop()

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.flush()
    input_video_path = temp_input.name

    try:
        classifier = load_classifier(str(MODEL_PATH), str(LABEL_MAP_PATH))
        detector   = load_detector()
    except Exception as exc:
        st.error(f"Lỗi khởi tạo tài nguyên: {exc}")
        st.stop()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Không thể mở video.")
        st.stop()

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    effective_fps             = src_fps / max(frame_skip, 1)
    out_fps                   = effective_fps if effective_fps > 1 else 25.0
    drowsy_threshold_frames   = max(1, int(drowsy_seconds_threshold   * effective_fps))
    recovery_threshold_frames = max(1, int(recovery_seconds_threshold * effective_fps))

    st.info(
        f"FPS nguồn: {src_fps:.2f} | Kích thước: {src_width}×{src_height} | "
        f"Tổng frame: {total_frames} | effective FPS: {effective_fps:.2f} | "
        f"Drowsy: {drowsy_threshold_frames} frames ({drowsy_seconds_threshold}s) | "
        f"Recovery: {recovery_threshold_frames} frames ({recovery_seconds_threshold}s)"
    )

    progress_bar = st.progress(0, text="Đang xử lý video…")
    output_video_path = OUTPUT_DIR / f"processed_{int(time.time())}.mp4"
    writer: cv2.VideoWriter | None = None

    score_buffer: deque[float] = deque(maxlen=smoothing_window)

    # ── Counters ──────────────────────────────────────────────────────────────
    # BẬT alarm : fused score cao LIÊN TỤC (nháy mắt → reset ngay về 0)
    # TẮT alarm : EAR > ngưỡng LIÊN TỤC   (nhắm lại → reset ngay về 0)
    drowsy_counter     = 0
    recovery_counter   = 0
    no_face_counter    = 0
    alarm_on           = False

    frame_idx            = 0
    processed_frames     = 0
    suspicious_frames    = 0
    alert_events_count   = 0
    total_no_face_frames = 0

    event_log: list[dict]      = []
    analytics_rows: list[dict] = []
    alert_started_at_frame: int | None = None
    alarm_intervals: list[tuple[float, float]] = []
    current_alarm_start: float | None = None

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
            scale  = display_width / float(w0)
            new_h  = max(1, int(h0 * scale))
            frame  = cv2.resize(frame, (display_width, new_h), interpolation=cv2.INTER_AREA)

            result = detector.process(
                frame, draw=True,
                draw_landmarks=draw_landmarks,
                draw_face_bbox=draw_face_bbox,
                draw_eye_boxes=draw_eye_boxes,
            )
            display = result.frame_bgr.copy()

            left_ear = right_ear = avg_ear = 0.0
            left_closed_prob = right_closed_prob = 0.0
            ear_score = cnn_score = fused_score = smooth_score = 0.0

            if not result.face_found:
                no_face_counter      += 1
                total_no_face_frames += 1
                drowsy_counter        = 0
                recovery_counter      = 0
                if alarm_on:
                    alarm_on = False
                    if current_alarm_start is not None:
                        alarm_intervals.append((current_alarm_start, frame_idx / src_fps))
                        current_alarm_start = None
                    if alert_started_at_frame is not None:
                        event_log.append({
                            "Thời điểm": format_seconds(frame_idx / src_fps),
                            "Sự kiện":   "Cảnh báo kết thúc (mất khuôn mặt)",
                        })
                        alert_started_at_frame = None

            else:
                no_face_counter = 0

                left_ear, right_ear, avg_ear = compute_average_ear(
                    result.left_eye_points, result.right_eye_points)
                left_pred  = classify_eye_state(classifier, result.left_eye_crop)
                right_pred = classify_eye_state(classifier, result.right_eye_crop)
                left_closed_prob  = classifier.get_closed_prob(left_pred)
                right_closed_prob = classifier.get_closed_prob(right_pred)

                ear_score, cnn_score, fused_score = fuse_eye_scores(
                    avg_ear=avg_ear,
                    left_closed_prob=left_closed_prob,
                    right_closed_prob=right_closed_prob,
                    alpha=alpha, ear_open=ear_open, ear_closed=ear_closed_thresh,
                )
                score_buffer.append(fused_score)
                smooth_score = float(np.mean(score_buffer)) if score_buffer else fused_score

                model_closed        = (left_closed_prob  >= model_closed_threshold or
                                       right_closed_prob >= model_closed_threshold)
                suspicious_now      = smooth_score >= suspicious_fused_threshold or model_closed
                alert_condition_now = (smooth_score >= alert_fused_threshold or
                                       (avg_ear <= ear_closed_thresh and model_closed))

                if suspicious_now:
                    suspicious_frames += 1

                # ════════════════════════════════════════════════════════════
                # COUNTER LOGIC CẢI TIẾN (CHỐNG RUNG / GIẢM PHẠT)
                # ════════════════════════════════════════════════════════════
                if not alarm_on:
                    if alert_condition_now:
                        drowsy_counter += 1
                    else:
                        # Mở mắt: giảm dần bộ đếm thay vì xóa trắng (cho phép chớp mắt nhanh)
                        drowsy_counter = max(0, drowsy_counter - 1)

                    if drowsy_counter >= drowsy_threshold_frames:
                        alarm_on               = True
                        alert_events_count    += 1
                        alert_started_at_frame = frame_idx
                        current_alarm_start    = frame_idx / src_fps
                        drowsy_counter         = 0
                        recovery_counter       = 0
                        event_log.append({
                            "Thời điểm": format_seconds(frame_idx / src_fps),
                            "Sự kiện":   "🚨 Cảnh báo buồn ngủ được kích hoạt",
                        })

                else:   # alert đang bật — kết hợp EAR và Smooth Score để hồi phục
                    recovery_condition_now = (
                        avg_ear > ear_closed_thresh and
                        smooth_score < suspicious_fused_threshold
                    )

                    if recovery_condition_now:
                        recovery_counter += 1
                    else:
                        # Nhắm lại: giảm dần thay vì reset về 0 (chống nhiễu frame)
                        recovery_counter = max(0, recovery_counter - 1)

                    if recovery_counter >= recovery_threshold_frames:
                        alarm_on         = False
                        drowsy_counter   = 0
                        recovery_counter = 0
                        if current_alarm_start is not None:
                            alarm_intervals.append((current_alarm_start, frame_idx / src_fps))
                            current_alarm_start = None
                        event_log.append({
                            "Thời điểm": format_seconds(frame_idx / src_fps),
                            "Sự kiện":   "✅ Đã hồi phục / Tắt cảnh báo",
                        })
                        alert_started_at_frame = None

            status_text, status_color = resolve_status_text(
                face_found=result.face_found, alert_on=alarm_on,
                fused_score=smooth_score,
                suspicious_threshold=suspicious_fused_threshold,
            )
            display = draw_status_ui(
                frame=display, status_text=status_text, status_color=status_color,
                fps=src_fps, avg_ear=avg_ear, ear_score=ear_score,
                cnn_score=cnn_score, fused_score=smooth_score,
                drowsy_counter=drowsy_counter, recovery_counter=recovery_counter,
                no_face_counter=no_face_counter,
                drowsy_threshold_frames=drowsy_threshold_frames,
                recovery_threshold_frames=recovery_threshold_frames,
                alert_on=alarm_on,
            )

            if writer is None:
                out_h, out_w = display.shape[:2]
                writer = create_video_writer(str(output_video_path), out_fps, out_w, out_h)
            writer.write(display)

            analytics_rows.append({
                "Frame":        frame_idx,
                "Thời điểm":   format_seconds(frame_idx / src_fps),
                "Khuôn mặt":   "Có" if result.face_found else "Không",
                "AVG EAR":     round(float(avg_ear), 4),
                "CNN Score":   round(float(cnn_score), 4),
                "Fused Score": round(float(smooth_score), 4),
                "Trạng thái":  status_text,
                "Cảnh báo":    "🔴" if alarm_on else "—",
            })

            pct = min(int(frame_idx / max(total_frames, 1) * 100), 100)
            progress_bar.progress(pct, text=f"Đang xử lý… {pct}%  (frame {frame_idx}/{total_frames})")

        cap.release()
        if writer is not None:
            writer.release()

        if alarm_on and current_alarm_start is not None:
            alarm_intervals.append((current_alarm_start, frame_idx / src_fps))

    except Exception as exc:
        cap.release()
        if writer is not None:
            writer.release()
        st.error(f"Lỗi xử lý video: {exc}")
        st.stop()

    # ── Re-encode H.264 ───────────────────────────────────────────────────────
    progress_bar.progress(100, text="⚙️ Đang chuyển mã H.264…")
    reencoded_path = OUTPUT_DIR / f"web_{output_video_path.stem}.mp4"
    encode_ok = reencode_video_h264(output_video_path, reencoded_path, fps=out_fps)

    if encode_ok:
        final_video_path = reencoded_path
    else:
        st.warning(
            "Re-encode H.264 thất bại. Để fix: cài ffmpeg tại ffmpeg.org rồi thêm vào PATH. "
            "Video vẫn được xuất — dùng nút **Tải về** để xem offline."
        )
        final_video_path = output_video_path

    # ── Mix alarm audio ───────────────────────────────────────────────────────
    if mix_alarm_audio and alarm_intervals and ALARM_PATH.exists() and encode_ok:
        progress_bar.progress(100, text="🔊 Đang mix tiếng cảnh báo…")
        audio_mixed_path = OUTPUT_DIR / f"final_{output_video_path.stem}.mp4"
        video_duration_sec = processed_frames / max(out_fps, 1e-6)

        if mix_alarm_into_video(
            final_video_path,
            ALARM_PATH,
            audio_mixed_path,
            alarm_intervals,
            video_duration_sec
        ):
            final_video_path = audio_mixed_path
        else:
            st.warning("Mix audio thất bại. Video sẽ không có tiếng cảnh báo.")

    progress_bar.progress(100, text="✅ Xử lý hoàn tất!")

    # ── Kết quả ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Kết quả phân tích")
    processed_duration_sec = processed_frames / max(out_fps, 1e-6)
    suspicious_ratio       = suspicious_frames / max(processed_frames, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frame đã xử lý",      processed_frames)
    col2.metric("Số lần cảnh báo",      alert_events_count)
    col3.metric("Tỷ lệ frame nghi ngờ", f"{suspicious_ratio * 100:.1f}%")
    col4.metric("Frame không có mặt",   total_no_face_frames)
    col5, col6 = st.columns(2)
    col5.metric("Thời lượng xử lý", format_seconds(processed_duration_sec))
    col6.metric("FPS nguồn",        f"{src_fps:.2f}")

    st.markdown("#### 📋 Nhật ký sự kiện cảnh báo")
    if event_log:
        st.dataframe(pd.DataFrame(event_log), use_container_width=True, hide_index=True)
    else:
        st.success("Không có sự kiện cảnh báo nào — tài xế tỉnh táo suốt video.")

    with st.expander("🔍 Bảng phân tích chi tiết từng frame", expanded=False):
        df_analytics = pd.DataFrame(analytics_rows)
        st.dataframe(df_analytics, use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️ Tải bảng phân tích (CSV)",
            data=df_analytics.to_csv(index=False).encode("utf-8"),
            file_name="analytics.csv", mime="text/csv",
        )

    st.markdown("---")
    st.subheader("🎬 Video kết quả")
    if final_video_path.exists():
        with open(final_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button(
            label="⬇️ Tải video đã xử lý",
            data=video_bytes,
            file_name=final_video_path.name,
            mime="video/mp4",
        )
    else:
        st.warning("Không tìm thấy video kết quả.")


# =========================
# Trang 3: Đánh giá & Hiệu năng
# =========================

def render_page_evaluation() -> None:
    st.title("📊 3. Đánh giá & Hiệu năng (Evaluation)")

    metrics    = load_metrics_dict()
    cm_df      = load_confusion_matrix_df()
    history_df = load_training_history_df()

    st.markdown("### 🏆 Các chỉ số Đánh giá (Test Set · 547 mẫu)")
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{metrics.get('accuracy',  0):.4f}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        col3.metric("Recall",    f"{metrics.get('recall',    0):.4f}")
        col4.metric("F1-score",  f"{metrics.get('f1_score',  0):.4f}")
    else:
        TP, FN, FP, TN = 349, 12, 0, 186
        total = TP + FP + FN + TN
        acc  = (TP + TN) / total
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1   = 2 * TP / (2 * TP + FP + FN)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{acc:.4f}",  delta="97.81%")
        col2.metric("Precision", f"{prec:.4f}", delta="Zero FP ✅")
        col3.metric("Recall",    f"{rec:.4f}")
        col4.metric("F1-score",  f"{f1:.4f}")

    st.warning("⚠️ Recall = 0.9668 (FN = 12). Precision = 1.0000 (FP = 0) — mọi cảnh báo đều chính xác.")
    st.markdown("**So sánh chỉ số theo lớp:**")
    plot_metrics_bar()

    report_data = {
        "Nhãn":      ["Closed_Eyes", "Open_Eyes", "Macro avg", "Weighted avg", "Accuracy"],
        "Precision": ["1.0000",      "0.9394",    "0.9697",    "0.9794",       "—"],
        "Recall":    ["0.9668",      "1.0000",    "0.9834",    "0.9781",       "—"],
        "F1-Score":  ["0.9831",      "0.9688",    "0.9759",    "0.9782",       "0.9781"],
        "Support":   ["361",         "186",       "547",       "547",          "547"],
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📈 Biểu đồ Kỹ thuật")
    c1, c2 = st.columns(2)
    with c1:
        cm_img_path = BASE_DIR / "anh.png" / "confusion_matrix.png"
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="Confusion Matrix", use_container_width=True)
        elif cm_df is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm_df.values, cmap="Blues")
            ax.set_xticks(range(len(cm_df.columns)))
            ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(cm_df.index)))
            ax.set_yticklabels(cm_df.index)
            ax.set_title("Confusion Matrix")
            for i in range(cm_df.shape[0]):
                for j in range(cm_df.shape[1]):
                    ax.text(j, i, int(cm_df.values[i, j]),
                            ha="center", va="center", fontweight="bold")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig); plt.close(fig)
        else:
            plot_confusion_matrix_custom()
    with c2:
        acc_loss_path = BASE_DIR / "anh.png" / "accuracy_loss.png"
        if acc_loss_path.exists():
            st.image(str(acc_loss_path), caption="Accuracy & Loss", use_container_width=True)
        elif history_df is not None and "epoch" in history_df.columns:
            cols = (["train_accuracy", "val_accuracy"]
                    if {"train_accuracy", "val_accuracy"}.issubset(history_df.columns)
                    else ["accuracy", "val_accuracy"])
            draw_history_chart(history_df, cols, "Accuracy theo Epoch", "Accuracy")
        else:
            plot_training_curves()

    if history_df is not None and "epoch" in history_df.columns:
        loss_cols = (["train_loss", "val_loss"]
                     if {"train_loss", "val_loss"}.issubset(history_df.columns)
                     else ["loss", "val_loss"])
        draw_history_chart(history_df, loss_cols, "Loss theo Epoch", "Loss")

    st.markdown("---")
    with st.expander("📋 Bảng chi tiết 15 Epochs", expanded=False):
        epochs_table = pd.DataFrame({
            "Epoch":      list(range(1, 16)),
            "Train Loss": [0.1123, 0.0348, 0.0203, 0.0178, 0.0148, 0.0103, 0.0107, 0.0070, 0.0086, 0.0067, 
                           0.0792, 0.0245, 0.0185, 0.0106, 0.0098],
            "Val Loss":   [0.0803, 0.0623, 0.0574, 0.0564, 0.0526, 0.0537, 0.0485, 0.0607, 0.0450, 0.0305, 
                           0.0575, 0.0540, 0.0643, 0.0767, 0.0772],
            "Train Acc":  [0.9672, 0.9913, 0.9948, 0.9947, 0.9946, 0.9963, 0.9968, 0.9980, 0.9965, 0.9981, 
                           0.9712, 0.9917, 0.9922, 0.9964, 0.9972],
            "Val Acc":    [0.9732, 0.9777, 0.9777, 0.9762, 0.9777, 0.9777, 0.9792, 0.9777, 0.9807, 0.9821, 
                           0.9792, 0.9792, 0.9792, 0.9762, 0.9762],
            "Ghi chú":    ["Khởi động", "", "", "", "", "", "", "", "", "✅ Best (98.21%)", 
                           "⚡ Fine-tune", "", "", "", "Final / Early Stop"],
        })
        st.dataframe(epochs_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💡 Phân tích sai số & Giải pháp")
    c_err, c_fix = st.columns(2)
    with c_err:
        st.error("**❌ 12 False Negatives**")
        st.markdown("- Mắt sụp mí / nhắm không hoàn toàn\n- Góc xoay đầu\n- Kính ánh chói")
    with c_fix:
        st.success("**✅ Hướng cải thiện**")
        st.markdown("- PERCLOS Metric\n- Data Augmentation\n- Temporal LSTM/GRU\n- Head Pose Estimation")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy",      "97.81%", delta="Tăng 2.56%")
    col2.metric("Precision (Closed)", "100.0%", delta="Zero FP ✅")
    col3.metric("Recall (Closed)",    "96.68%")
    col4.metric("Model Params",       "~2.3M")


# =========================
# App router
# =========================
st.sidebar.title("Điều hướng")
page = st.sidebar.radio(
    "Chọn trang",
    ["1. Giới thiệu & EDA", "2. Triển khai mô hình", "3. Đánh giá & Hiệu năng"],
)

if page == "1. Giới thiệu & EDA":
    render_page_eda()
elif page == "2. Triển khai mô hình":
    render_page_deploy()
else:
    render_page_evaluation()

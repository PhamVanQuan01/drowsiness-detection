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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
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
    ear_score = ear_to_closed_score(ear=avg_ear, ear_open=ear_open, ear_closed=ear_closed)
    cnn_score = (left_closed_prob + right_closed_prob) / 2.0
    fused_score = alpha * cnn_score + (1.0 - alpha) * ear_score
    return float(ear_score), float(cnn_score), float(fused_score)


def file_to_base64(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_browser_alarm(placeholder, alarm_wav_path, should_play, key="browser_alarm"):
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
        if (audio) {{ audio.play().catch((e) => {{ console.log("Autoplay bị chặn:", e); }}); }}
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
    frame, status_text, status_color, fps, avg_ear,
    ear_score, cnn_score, fused_score,
    drowsy_counter, recovery_counter, no_face_counter,
    drowsy_threshold_frames, recovery_threshold_frames, alert_on,
) -> np.ndarray:
    display = draw_header_panel(frame)
    h, w = display.shape[:2]
    font_scale = max(w / 1100.0, 0.55)
    thickness = max(int(font_scale * 2), 1)
    cv2.putText(display, status_text, (int(w*0.03), int(h*0.07)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness+1, cv2.LINE_AA)
    if alert_on:
        cv2.rectangle(display, (10, 10), (w-10, h-10), (0, 0, 255), 3)
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(display, "DROWSY ALERT!", (int(w*0.03), int(h*0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.95, (0,0,255), thickness+1, cv2.LINE_AA)
    bar_x1, bar_y1 = int(w*0.03), int(h*0.11)
    bar_x2, bar_y2 = int(w*0.35), int(h*0.145)
    ratio = min(drowsy_counter / max(drowsy_threshold_frames, 1), 1.0)
    fill_x = int(bar_x1 + (bar_x2 - bar_x1) * ratio)
    cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (255,255,255), 2)
    bar_color = (0, int(255*(1-ratio)), int(255*ratio))
    cv2.rectangle(display, (bar_x1, bar_y1), (fill_x, bar_y2), bar_color, -1)
    cv2.putText(display, "Drowsiness level", (bar_x1, bar_y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.45, (255,255,255), 1, cv2.LINE_AA)
    right_x = int(w*0.70)
    start_y = int(h*0.06)
    line_h = int(max(22, h*0.032))
    stats = [
        ("===== SYSTEM =====", (255,255,255)), (f"FPS: {fps:.2f}", (0,255,0)), ("", (255,255,255)),
        ("===== SCORES =====", (255,255,255)), (f"AVG EAR: {avg_ear:.3f}", (255,255,255)),
        (f"EAR Score: {ear_score:.3f}", (255,255,255)), (f"CNN Score: {cnn_score:.3f}", (255,255,255)),
        (f"Fused Score: {fused_score:.3f}", (0,200,255)), ("", (255,255,255)),
        ("===== COUNTERS =====", (255,255,255)),
        (f"Drowsy: {drowsy_counter}/{drowsy_threshold_frames}", (255,255,255)),
        (f"Recovery: {recovery_counter}/{recovery_threshold_frames}", (255,255,255)),
        (f"No face: {no_face_counter}", (255,255,255)),
    ]
    for i, (text, color) in enumerate(stats):
        y = start_y + i * line_h
        if y < h - 20:
            cv2.putText(display, text, (right_x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale*0.42, color, 1, cv2.LINE_AA)
    return display


def resolve_status_text(face_found, alert_on, fused_score, suspicious_threshold):
    if not face_found:
        return "NO FACE DETECTED", (0, 200, 255)
    if alert_on:
        return "DROWSY ALERT", (0, 0, 255)
    if fused_score >= suspicious_threshold:
        return "SUSPICIOUS", (0, 200, 255)
    return "AWAKE", (0, 255, 0)


def create_video_writer(output_path, fps, width, height):
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
# Chart helpers for EDA & Eval
# =========================

def plot_label_distribution():
    """Biểu đồ cột phân phối nhãn toàn bộ dataset."""
    labels = ["Closed_Eyes", "Open_Eyes"]
    counts = [2726, 2726]
    colors = ["#e74c3c", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
    ax.set_title("Phân phối nhãn (Label Distribution)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Số lượng mẫu", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{count:,}\n({count/sum(counts)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_split_distribution():
    """Biểu đồ tròn phân chia train/val/test."""
    splits = ["Train (77.6%)", "Val (12.3%)", "Test (10.1%)"]
    sizes = [4233, 672, 547]
    colors = ["#3498db", "#f39c12", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=splits, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    autotexts[2].set_text("10.1%")
    ax.set_title("Phân chia dữ liệu Train / Val / Test\n(Tổng: 5,452 ảnh)", fontsize=12, fontweight="bold", pad=12)
    centre_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(centre_circle)
    ax.text(0, 0, "5,452\nảnh", ha="center", va="center", fontsize=11, fontweight="bold", color="#555")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_split_label_breakdown():
    """Biểu đồ cột nhóm: phân phối nhãn theo từng tập."""
    splits = ["Train", "Val", "Test"]
    closed = [2029, 336, 361]
    open_e = [2204, 336, 186]
    x = np.arange(len(splits))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width/2, closed, width, label="Closed_Eyes", color="#e74c3c", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, open_e, width, label="Open_Eyes",   color="#2ecc71", alpha=0.85, edgecolor="white")
    ax.set_title("Phân phối nhãn theo từng tập dữ liệu", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylabel("Số lượng mẫu", fontsize=10)
    ax.legend(fontsize=10)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_ear_distribution():
    """Phân phối EAR theo nhãn (Gaussian ước tính)."""
    x = np.linspace(0, 0.5, 500)
    closed_dist = np.exp(-0.5 * ((x - 0.10) / 0.025)**2)
    open_dist   = np.exp(-0.5 * ((x - 0.30) / 0.040)**2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, closed_dist, alpha=0.45, color="#e74c3c", label="Closed_Eyes (μ≈0.10)")
    ax.fill_between(x, open_dist,   alpha=0.45, color="#2ecc71", label="Open_Eyes (μ≈0.30)")
    ax.plot(x, closed_dist, color="#c0392b", linewidth=1.8)
    ax.plot(x, open_dist,   color="#27ae60", linewidth=1.8)
    ax.axvline(0.18, color="#f39c12", linewidth=2, linestyle="--", label="Ngưỡng EAR = 0.18")
    ax.axvline(0.30, color="#3498db", linewidth=1.5, linestyle=":",  label="EAR mở chuẩn = 0.30")
    ax.set_title("Phân phối EAR theo nhãn (ước tính)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Giá trị EAR", fontsize=11)
    ax.set_ylabel("Mật độ xác suất (chuẩn hóa)", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_ear_diagram():
    """Sơ đồ 6 điểm landmark EAR."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Eye shape
    eye_x = np.linspace(1.5, 7.5, 200)
    eye_y_top = 2.0 + 0.9 * np.sin(np.pi * (eye_x - 1.5) / 6.0)
    eye_y_bot = 2.0 - 0.9 * np.sin(np.pi * (eye_x - 1.5) / 6.0)
    ax.fill_between(eye_x, eye_y_bot, eye_y_top, color="#2c3e50", alpha=0.5)
    ax.plot(eye_x, eye_y_top, color="#00bcd4", linewidth=1.5)
    ax.plot(eye_x, eye_y_bot, color="#00bcd4", linewidth=1.5)

    # 6 landmarks
    pts = {
        "p1": (1.5, 2.0), "p2": (3.0, 2.75), "p3": (5.5, 2.75),
        "p4": (7.5, 2.0), "p5": (5.5, 1.25), "p6": (3.0, 1.25),
    }
    corner_color = "#f39c12"
    lid_color    = "#00e5ff"
    for name, (px, py) in pts.items():
        color = corner_color if name in ("p1", "p4") else lid_color
        ax.plot(px, py, "o", color=color, markersize=9, zorder=5)
        offset = {"p1": (-0.35, 0), "p2": (0, 0.25), "p3": (0, 0.25),
                  "p4": (0.25, 0), "p5": (0, -0.32), "p6": (0, -0.32)}
        dx, dy = offset[name]
        ax.text(px + dx, py + dy, name, color="white", fontsize=9, fontweight="bold", ha="center")

    # Dimension lines
    # Vertical p2-p6
    ax.annotate("", xy=(3.0, 1.25), xytext=(3.0, 2.75),
                arrowprops=dict(arrowstyle="<->", color="#2ecc71", lw=1.5))
    ax.text(2.55, 2.0, "||p2-p6||", color="#2ecc71", fontsize=8, ha="center", rotation=90)
    # Vertical p3-p5
    ax.annotate("", xy=(5.5, 1.25), xytext=(5.5, 2.75),
                arrowprops=dict(arrowstyle="<->", color="#2ecc71", lw=1.5))
    ax.text(5.95, 2.0, "||p3-p5||", color="#2ecc71", fontsize=8, ha="center", rotation=90)
    # Horizontal p1-p4
    ax.annotate("", xy=(7.5, 1.6), xytext=(1.5, 1.6),
                arrowprops=dict(arrowstyle="<->", color=corner_color, lw=1.5))
    ax.text(4.5, 1.35, "||p1-p4||", color=corner_color, fontsize=8, ha="center")

    # Formula box
    formula = "EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)"
    ax.text(4.5, 0.45, formula, color="#ecf0f1", fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1b2a", edgecolor="#00bcd4", linewidth=1.5))

    ax.set_title("Sơ đồ 6 điểm landmark tính EAR", color="white", fontsize=11, fontweight="bold", pad=8)
    st.pyplot(fig)
    plt.close(fig)


def plot_mobilenet_architecture():
    """Sơ đồ kiến trúc pipeline MobileNetV2."""
    blocks = [
        ("Input\n224×224×3",        "#8e44ad"),
        ("Normalize\n& Data Aug",   "#2980b9"),
        ("MobileNetV2\n(ImageNet)", "#2980b9"),
        ("GlobalAvg\nPool2D",       "#16a085"),
        ("Dropout",                 "#16a085"),
        ("Dense(1)\n(Sigmoid)",     "#16a085"),
        ("Xác suất\nClosed/Open",   "#27ae60"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    n = len(blocks)
    for i, (label, color) in enumerate(blocks):
        x = i / (n - 1)
        rect = mpatches.FancyBboxPatch(
            (x - 0.055, 0.15), 0.11, 0.70,
            boxstyle="round,pad=0.02",
            facecolor=color + "33", edgecolor=color, linewidth=1.8,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(x, 0.50, label, transform=ax.transAxes, color="white",
                fontsize=8, ha="center", va="center", multialignment="center")
        if i < n - 1:
            ax.annotate("", xy=((i+1)/(n-1) - 0.057, 0.50),
                        xytext=(x + 0.057, 0.50),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#95a5a6", lw=1.2))
    ax.set_title("Kiến trúc Pipeline MobileNetV2 (Transfer Learning từ ImageNet)",
                 color="white", fontsize=11, fontweight="bold", pad=10)
    st.pyplot(fig)
    plt.close(fig)


def plot_fusion_diagram():
    """Sơ đồ luồng Fusion Score."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    ax.set_facecolor("#0f1923")
    fig.patch.set_facecolor("#0f1923")

    def box(cx, cy, text, color, width=0.14, height=0.32):
        rect = mpatches.FancyBboxPatch(
            (cx - width/2, cy - height/2), width, height,
            boxstyle="round,pad=0.015",
            facecolor=color + "22", edgecolor=color, linewidth=1.8,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, transform=ax.transAxes,
                color="white", fontsize=8, ha="center", va="center", multialignment="center")

    def arrow(x1, y1, x2, y2, label="", color="#95a5a6"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.06, label, transform=ax.transAxes,
                    color=color, fontsize=7, ha="center")

    # Inputs
    box(0.05, 0.75, "Left Eye\nCrop", "#9b59b6")
    box(0.05, 0.25, "Right Eye\nCrop", "#9b59b6")
    box(0.20, 0.50, "MediaPipe\nFace Mesh", "#2980b9")

    # CNN scores
    box(0.35, 0.75, "CNN Output\nP(closed)_L", "#e67e22")
    box(0.35, 0.25, "CNN Output\nP(closed)_R", "#e67e22")

    # EAR
    box(0.35, 0.50, "EAR\nNormalized", "#16a085")

    # Average & Fusion
    box(0.52, 0.65, "CNN_score\n= avg(L,R)", "#e67e22")
    box(0.68, 0.50, "Fused Score\n= α*CNN\n+(1-α)*EAR", "#c0392b")

    # Smoothing
    box(0.83, 0.50, "Temporal\nSmoothing\n(Mean N frames)", "#3498db")

    # Decision
    box(0.96, 0.75, "AWAKE\n✅", "#27ae60")
    box(0.96, 0.50, "SUSPICIOUS\n⚠️", "#f39c12")
    box(0.96, 0.25, "ALERT\n🚨", "#e74c3c")

    # Arrows
    arrow(0.11, 0.75, 0.16, 0.62)
    arrow(0.11, 0.25, 0.16, 0.38)
    arrow(0.24, 0.62, 0.30, 0.55)
    arrow(0.24, 0.68, 0.30, 0.75)
    arrow(0.24, 0.38, 0.30, 0.25)
    arrow(0.40, 0.75, 0.46, 0.70)
    arrow(0.40, 0.25, 0.46, 0.60)
    arrow(0.40, 0.50, 0.62, 0.52)
    arrow(0.58, 0.65, 0.62, 0.58)
    arrow(0.74, 0.50, 0.77, 0.50)
    arrow(0.89, 0.58, 0.91, 0.73, "< threshold", "#27ae60")
    arrow(0.89, 0.50, 0.91, 0.50, "≥ thresh", "#f39c12")
    arrow(0.89, 0.42, 0.91, 0.27, "≥ th + time", "#e74c3c")

    ax.set_title("Sơ đồ luồng Fusion Score (EAR + CNN → Smoothing → Quyết định)",
                 color="white", fontsize=11, fontweight="bold", pad=8)
    st.pyplot(fig)
    plt.close(fig)


def plot_state_machine():
    """Sơ đồ máy trạng thái cảnh báo."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_facecolor("#0f1923")
    fig.patch.set_facecolor("#0f1923")

    states = [
        (1.8, 2.0, "AWAKE\n😊",        "#27ae60", 0.85),
        (5.0, 2.0, "SUSPICIOUS\n😐",   "#f39c12", 0.85),
        (8.2, 2.0, "DROWSY ALERT\n🚨", "#e74c3c", 0.95),
    ]
    for cx, cy, label, color, r in states:
        circle = plt.Circle((cx, cy), r, facecolor=color+"22",
                             edgecolor=color, linewidth=2.2)
        ax.add_patch(circle)
        ax.text(cx, cy, label, color="white", fontsize=9.5,
                ha="center", va="center", fontweight="bold", multialignment="center")

    # Forward arrows
    ax.annotate("", xy=(4.12, 2.25), xytext=(2.65, 2.25),
                arrowprops=dict(arrowstyle="-|>", color="#f39c12", lw=1.8))
    ax.text(3.38, 2.55, "smooth_score\n≥ suspicious", color="#f39c12", fontsize=7.5, ha="center")

    ax.annotate("", xy=(7.27, 2.25), xytext=(5.88, 2.25),
                arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1.8))
    ax.text(6.58, 2.55, "alert_condition_now\nduy trì N frame", color="#e74c3c", fontsize=7.5, ha="center")

    # Backward arrows
    ax.annotate("", xy=(2.65, 1.75), xytext=(4.12, 1.75),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5))
    ax.text(3.38, 1.45, "smooth_score\n< suspicious", color="#27ae60", fontsize=7.5, ha="center")

    ax.annotate("", xy=(5.88, 1.75), xytext=(7.27, 1.75),
                arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.5))
    ax.text(6.58, 1.45, "alert_condition_now=False\nđủ M frame", color="#27ae60", fontsize=7.5, ha="center")

    # No face reset
    ax.annotate("", xy=(1.8, 1.0), xytext=(5.0, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#95a5a6", lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(3.38, 0.45, "No face detected → reset counters, stop alarm", color="#95a5a6", fontsize=7.5, ha="center")

    ax.set_title("Máy trạng thái cảnh báo buồn ngủ", color="white",
                 fontsize=11, fontweight="bold", pad=8)
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion_matrix_custom():
    """Vẽ confusion matrix từ số liệu thực tế (TP=335, FN=26, FP=0, TN=186)."""
    cm = np.array([[335, 26], [0, 186]])
    labels = ["Closed_Eyes", "Open_Eyes"]
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Test Set (547 mẫu)", fontsize=12, fontweight="bold", pad=12)

    cell_labels = [
        ["TP = 335\n(61.2%)", "FN = 26\n(4.8%)"],
        ["FP = 0\n(0.0%)", "TN = 186\n(34.0%)"],
    ]
    colors_text = [["white", "#1a3a6e"], ["#1a3a6e", "white"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=colors_text[i][j])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_training_curves():
    """Vẽ Loss và Accuracy theo 14 epoch từ số liệu thực tế."""
    epochs = list(range(1, 15))
    train_loss = [0.112, 0.026, 0.019, 0.017, 0.013, 0.010, 0.008, 0.007, 0.006, 0.006, 0.079, 0.025, 0.019, 0.016]
    val_loss   = [0.080, 0.070, 0.064, 0.058, 0.057, 0.054, 0.055, 0.061, 0.057, 0.031, 0.058, 0.054, 0.054, 0.065]
    train_acc  = [0.9655, 0.9947, 0.9962, 0.9962, 0.9968, 0.9972, 0.9973, 0.9982, 0.9979, 0.9979, 0.9717, 0.9920, 0.9960, 0.9949]
    val_acc    = [0.9730, 0.9775, 0.9775, 0.9762, 0.9775, 0.9775, 0.9775, 0.9775, 0.9775, 0.9821, 0.9800, 0.9786, 0.9786, 0.9762]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs, train_loss, "o-", color="#3498db", linewidth=2, markersize=5, label="Train Loss")
    ax1.plot(epochs, val_loss,   "s-", color="#e67e22", linewidth=2, markersize=5, label="Val Loss")
    ax1.axvline(11, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.text(11.2, max(train_loss)*0.9, "LR reset\n(fine-tune)", color="#e74c3c", fontsize=8)
    ax1.set_title("Loss theo Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    # Accuracy
    ax2.plot(epochs, train_acc, "o-", color="#27ae60", linewidth=2, markersize=5, label="Train Accuracy")
    ax2.plot(epochs, val_acc,   "s-", color="#8e44ad", linewidth=2, markersize=5, label="Val Accuracy")
    best_epoch = val_acc.index(max(val_acc)) + 1
    ax2.axvline(best_epoch, color="#f39c12", linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.text(best_epoch + 0.2, min(val_acc) + 0.001,
             f"Best epoch {best_epoch}\nVal={max(val_acc):.4f}", color="#f39c12", fontsize=8)
    ax2.set_title("Accuracy theo Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f9fa")

    plt.suptitle("Quá trình huấn luyện MobileNetV2 — 14 Epochs", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_metrics_bar():
    """Biểu đồ cột so sánh các chỉ số per-class."""
    metrics_data = {
        "Precision": {"Closed_Eyes": 1.0000, "Open_Eyes": 0.8776, "Macro avg": 0.9388},
        "Recall":    {"Closed_Eyes": 0.9281, "Open_Eyes": 1.0000, "Macro avg": 0.9640},
        "F1-Score":  {"Closed_Eyes": 0.9626, "Open_Eyes": 0.9350, "Macro avg": 0.9488},
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    width = 0.22
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    classes = ["Closed_Eyes", "Open_Eyes", "Macro avg"]
    for idx, (metric, vals) in enumerate(metrics_data.items()):
        ys = [vals[c] for c in classes]
        bars = ax.bar(x + idx * width, ys, width, label=metric, color=colors[idx], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0.85, 1.04)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("So sánh Precision / Recall / F1 theo lớp", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================
# Pages
# =========================

def render_page_eda() -> None:
    st.title("🚀 1. Giới thiệu & Khám phá dữ liệu")
    st.markdown("---")

    st.markdown(
        "<h3 style='text-align:center;color:#1f77b4;'>"
        " ĐỀ TÀI: PHÁT HIỆN TRẠNG THÁI BUỒN NGỦ CỦA TÀI XẾ BẰNG MÔ HÌNH "
        "MOBILENETV2 KẾT HỢP CHỈ SỐ EAR NHẰM CẢNH BÁO KỊP THỜI"
        "</h3>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ---- Thông tin sinh viên + mô tả ----
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info(
            "**Thông tin sinh viên**\n\n"
            "🎓 **Họ và tên:** Phạm Văn Quân\n\n"
            "🆔 **Mã sinh viên:** 22T1020347"
        )
    with col2:
        st.success("**Giới thiệu**\n\n"
            "Tai nạn giao thông do buồn ngủ là một trong những nguyên nhân hàng đầu gây ra thiệt hại nghiêm trọng. "
            "bài toán này nhằm **theo dõi và phân tích chỉ số khía cạnh mắt (EAR)** của tài xế qua camera theo thời gian thực "
            "và sẽ lập tức phát tín hiệu cảnh báo (âm thanh ) khi tài xế ngủ gật."
        )

    st.markdown("---")

    # ==========================================
    # LÝ THUYẾT CHI TIẾT
    # ==========================================
    st.markdown("### 🧠 Lý thuyết Mô Hình & Phương Pháp Nhận Diện")

    # ---------- 1. EAR ----------
    with st.expander("📌 1. Trích xuất đặc trưng hình học (Geometric Features) với MediaPipe & EAR", expanded=True):
        c_left, c_right = st.columns([1.1, 1])
        with c_left:
            st.markdown("""
**MediaPipe Face Mesh** cung cấp **468 điểm mốc (landmarks) 3D** trên khuôn mặt với tốc độ thực,
bao phủ chính xác đặc điểm quanh mắt (6 điểm mỗi mắt).

**Eye Aspect Ratio (EAR)** là chỉ số hình học tính tỷ lệ giữa chiều cao và chiều rộng của mắt:

$$EAR = \\frac{\\|p_2 - p_6\\| + \\|p_3 - p_5\\|}{2 \\times \\|p_1 - p_4\\|}$$

- **p1, p4**: góc trái và phải mắt
- **p2, p3**: điểm mí trên
- **p5, p6**: điểm mí dưới

| Trạng thái | Giá trị EAR |
|---|---|
| Mắt mở bình thường | 0.25 – 0.35 |
| Vùng nghi ngờ | 0.18 – 0.25 |
| Mắt nhắm (buồn ngủ) | < 0.18 |

Hệ thống tính **avg_EAR = (EAR_trái + EAR_phải) / 2** để tăng độ ổn định.
""")
        with c_right:
            plot_ear_diagram()

    # ---------- 2. MobileNetV2 ----------
    with st.expander("📌 2. Phân loại trạng thái mắt (Deep Learning) bằng MobileNetV2", expanded=True):
        st.markdown("""
**MobileNetV2** được sử dụng làm backbone trích xuất đặc trưng cho ảnh vùng mắt. Mô hình tận dụng trọng số pretrained từ ImageNet để tăng tốc độ hội tụ và giảm yêu cầu dữ liệu huấn luyện. Trong hệ thống này, ảnh mắt sau khi được crop và resize về kích thước 224×224 sẽ được đưa qua MobileNetV2, sau đó qua lớp GlobalAveragePooling2D, Dropout và lớp Dense đầu ra 1 nút với hàm sigmoid để thực hiện phân loại nhị phân trạng thái mắt mở hoặc nhắm.

**Về cấu trúc mạng gốc, MobileNetV2 chuyên biệt cho thiết bị nhúng:**
- **Inverted Residual Blocks**: Mở rộng kênh → Depthwise Conv → Thu hẹp. Giúp giảm ~8–9× tham số so với VGG16.
- **Linear Bottlenecks**: Không dùng hàm kích hoạt phi tuyến (ReLU) ở lớp cuối của block để tránh mất mát thông tin.
""")
        plot_mobilenet_architecture()

        st.markdown("""

```

""")

    # ---------- 3. Fusion ----------
    with st.expander("📌 3. Thuật toán kết hợp Fusion Score (EAR + CNN)", expanded=True):
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("""
Hệ thống kết hợp hai nguồn thông tin gồm đặc trưng hình học EAR và xác suất mắt nhắm từ mô hình MobileNetV2. Sau khi tính toán ear_score và cnn_score, hai giá trị này được kết hợp bằng phép nội suy có trọng số để tạo ra fused_score. Giá trị này tiếp tục được làm mượt theo thời gian nhằm giảm nhiễu giữa các frame liên tiếp. Cuối cùng, hệ thống sử dụng ngưỡng kết hợp với bộ đếm thời gian để xác định trạng thái tỉnh táo, nghi ngờ hoặc buồn ngủ, đảm bảo cảnh báo không bị kích hoạt sai do biến động tức thời.

**Chi tiết thuật toán:**
1. **EAR → Score**: Chuyển đổi EAR hình học sang thang điểm [0,1].
2. **CNN → Score**: Lấy giá trị thực tế xác suất nhắm mắt từ hai mắt: $cnn\\_score = (P\\_closed(L) + P\\_closed(R)) / 2$.
3. **Weighting (α)**: Kết hợp tuyến tính $fused\\_score = \\alpha \\cdot cnn + (1-\\alpha) \\cdot ear$.
4. **Smoothing**: Lấy trung bình trượt (Moving Average) qua $N$ frame để tạo ra điểm số ổn định.
5. **Validation**: Kích hoạt cảnh báo khi điểm số duy trì trên ngưỡng trong một khoảng thời gian xác định (bộ đếm frame).
""")
        with c_right:
            plot_fusion_diagram()

    # ---------- 4. State Machine ----------
    with st.expander("📌 4. Cơ chế cảnh báo dựa trên bộ đếm thời gian (Counter-based Alert)", expanded=False):
        st.markdown("""
Hệ thống không kích hoạt cảnh báo chỉ từ một frame đơn lẻ mà sử dụng bộ đếm theo thời gian. Khi điều kiện cảnh báo tức thời (`alert_condition_now`) được thỏa mãn liên tiếp trong đủ số frame quy đổi từ thời gian, `drowsy_counter` sẽ đạt ngưỡng và kích hoạt báo động. Ngược lại, khi điều kiện cảnh báo không còn thỏa mãn, `recovery_counter` tăng dần để xác nhận tài xế đã hồi phục trước khi tắt báo động. Trường hợp không phát hiện khuôn mặt, hệ thống sẽ reset các bộ đếm và tắt cảnh báo ngay để tránh suy luận sai.

**Thông số tính toán ngưỡng frame:**
```python
effective_fps = src_fps / frame_skip
drowsy_threshold_frames = int(drowsy_seconds_threshold * effective_fps)
recovery_threshold_frames = int(recovery_seconds_threshold * effective_fps)
```
- **Drowsy Counter**: Tăng khi `alert_condition_now` là True.
- **Recovery Counter**: Tăng khi `alert_condition_now` là False.
- **No-face**: Tắt alarm ngay lập tức và reset cả 2 bộ đếm.
""")
        plot_state_machine()

    # ==========================================
    # DỮ LIỆU THÔ
    # ==========================================
    st.markdown("---")
    st.markdown("### 📁 Cấu trúc Dataset (Kaggle) & Dữ liệu thô")

    st.info(
        "Dataset lấy từ **Kaggle** gồm **3 tập dữ liệu riêng biệt** (train / val / test). "
        "Mỗi tập chứa **2 thư mục ảnh**: `Closed_Eyes/` và `Open_Eyes/`. "
        "Tất cả ảnh đã được chuẩn hóa về kích thước **224 × 224 px**."
    )

    st.code(
        "dataset/\n"
        "├── train/\n"
        "│   ├── Closed_Eyes/   (2029 ảnh)\n"
        "│   └── Open_Eyes/     (2204 ảnh)\n"
        "├── val/\n"
        "│   ├── Closed_Eyes/   (336 ảnh)\n"
        "│   └── Open_Eyes/     (336 ảnh)\n"
        "└── test/\n"
        "    ├── Closed_Eyes/   (361 ảnh)\n"
        "    └── Open_Eyes/     (186 ảnh)",
        language="text",
    )


    # ==========================================
    # BIỂU ĐỒ PHÂN TÍCH
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Biểu đồ phân bố dữ liệu")

    c1, c2 = st.columns(2)
    with c1:
        df_meta_loaded = load_metadata_df()
        if df_meta_loaded is not None and "label" in df_meta_loaded.columns:
            draw_simple_bar_chart(
                df_meta_loaded["label"].value_counts(),
                "Phân phối nhãn (Label Distribution)", "Nhãn", "Số lượng mẫu",
            )
        else:
            plot_label_distribution()

    with c2:
        if df_meta_loaded is not None and "split" in df_meta_loaded.columns:
            draw_simple_bar_chart(
                df_meta_loaded["split"].value_counts(),
                "Tập Huấn luyện / Đánh giá (Train/Val/Test)", "Tập dữ liệu", "Số lượng",
            )
        else:
            plot_split_distribution()

    st.markdown("**Phân phối nhãn chi tiết theo từng tập:**")
    c3, c4 = st.columns(2)
    with c3:
        plot_split_label_breakdown()
    with c4:
        plot_ear_distribution()

    # ==========================================
    # NHẬN XÉT
    # ==========================================
    st.markdown("---")
    st.markdown("### 💡 Nhận xét dữ liệu")
    st.success(
        "✅ **Dữ liệu cân bằng nhãn:** Hai lớp Closed_Eyes và Open_Eyes phân bố gần như đồng đều (~50–50), "
        "giúp mô hình tránh bias và học tốt hơn."
    )
    st.success(
        "✅ **Kích thước đồng nhất:** Tất cả ảnh được chuẩn hóa về 224×224 px, "
        "phù hợp với input của MobileNetV2, giúp quá trình huấn luyện ổn định."
    )
    st.info(
        "🔑 **Đặc trưng quan trọng:** (1) trạng thái mắt mở/nhắm, "
        "(2) hình dạng mắt theo thời gian (EAR), "
        "(3) chất lượng crop vùng mắt từ FaceMesh, "
        "(4) ánh sáng và độ tương phản."
    )
    st.info(
        "📌 **Phân tách EAR:** Hai lớp có phân bố EAR khác biệt rõ, tuy nhiên tồn tại vùng chồng lấp (overlap) "
        "khiến EAR đơn lẻ không đủ phân biệt. Vì vậy cần kết hợp thêm mô hình CNN để tăng độ chính xác, "
        "dẫn đến việc sử dụng Fusion Score."
    )


# =========================
# Trang 2: Triển khai mô hình  (KHÔNG THAY ĐỔI)
# =========================

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
                frame, draw=True,
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
                    alpha=alpha, ear_open=ear_open, ear_closed=ear_closed,
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
                face_found=result.face_found, alert_on=alarm_on,
                fused_score=smooth_score, suspicious_threshold=suspicious_fused_threshold,
            )

            now = time.time()
            dt = max(now - prev_display_time, 1e-6)
            measured_fps = 1.0 / dt
            prev_display_time = now

            display = draw_status_ui(
                frame=display, status_text=status_text, status_color=status_color,
                fps=measured_fps, avg_ear=avg_ear, ear_score=ear_score,
                cnn_score=cnn_score, fused_score=smooth_score,
                drowsy_counter=drowsy_counter, recovery_counter=recovery_counter,
                no_face_counter=no_face_counter,
                drowsy_threshold_frames=drowsy_threshold_frames,
                recovery_threshold_frames=recovery_threshold_frames,
                alert_on=alarm_on,
            )

            render_browser_alarm(
                placeholder=audio_placeholder, alarm_wav_path=ALARM_PATH,
                should_play=alarm_on, key="drowsy_alarm_audio",
            )

            if save_output_video and writer is None:
                out_h, out_w = display.shape[:2]
                writer = create_video_writer(str(output_video_path),
                                             fps=src_fps if src_fps > 1 else 25.0,
                                             width=out_w, height=out_h)
            if writer is not None:
                writer.write(display)

            should_update_ui = processed_frames == 1 or (processed_frames % ui_update_every == 0)
            if should_update_ui:
                frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                progress_ratio = frame_idx / max(total_frames, 1)
                progress_placeholder.progress(
                    min(int(progress_ratio * 100), 100),
                    text=f"Đang xử lý frame {frame_idx}/{total_frames}",
                )
                stats_placeholder.json({
                    "frame_index": frame_idx, "processed_frames": processed_frames,
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
                })

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
            label="Tải video đã xử lý", data=video_bytes,
            file_name=output_video_path.name, mime="video/mp4",
        )


# =========================
# Trang 3: Đánh giá & Hiệu năng
# =========================

def render_page_evaluation() -> None:
    st.title("📊 3. Đánh giá & Hiệu năng (Evaluation)")
    st.write(
        
    )

    # ---- Load data files ----
    metrics = load_metrics_dict()
    cm_df = load_confusion_matrix_df()
    history_df = load_training_history_df()

    # ==========================================
    # METRICS
    # ==========================================
    st.markdown("### 🏆 Các chỉ số Đánh giá Kỹ thuật (Test Set · 547 mẫu)")

    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{metrics.get('accuracy',  0):.4f}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        col3.metric("Recall",    f"{metrics.get('recall',    0):.4f}")
        col4.metric("F1-score",  f"{metrics.get('f1_score',  0):.4f}")
    else:
        # Hiển thị từ confusion matrix thực tế
        TP, FN, FP, TN = 335, 26, 0, 186
        total = TP + FP + FN + TN
        acc  = (TP + TN) / total
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1   = 2 * TP / (2 * TP + FP + FN)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{acc:.4f}", delta="95.25%")
        col2.metric("Precision", f"{prec:.4f}", delta="Zero FP báo nhầm ✅")
        col3.metric("Recall",    f"{rec:.4f}")
        col4.metric("F1-score",  f"{f1:.4f}")

    st.warning(
        "⚠️ **Recall (Closed_Eyes) = 0.9280:** Mô hình có bỏ sót khoảng 7% trường hợp nhắm mắt (FN = 26). "
        "Tuy nhiên, **Precision = 1.0000** (FP = 0), nghĩa là mọi cảnh báo phát ra đều chính xác 100%. "
        "Trong các hệ thống thực tế có kết hợp với tính năng luồng thời gian (Temporal), cảnh báo có thể bù lại cho độ trễ nhỏ."
    )

    # ---- Per-class metrics bar ----
    st.markdown("**So sánh chỉ số theo lớp:**")
    plot_metrics_bar()

    # Classification report table
    report_data = {
        "Nhãn":       ["Closed_Eyes", "Open_Eyes", "Macro avg", "Weighted avg", "Accuracy"],
        "Precision":  ["1.0000",      "0.8774",    "0.9387",    "0.9583",       "—"],
        "Recall":     ["0.9280",      "1.0000",    "0.9640",    "0.9525",       "—"],
        "F1-Score":   ["0.9626",      "0.9347",    "0.9487",    "0.9531",       "0.9525"],
        "Support":    ["361",         "186",       "547",       "547",          "547"],
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==========================================
    # BIỂU ĐỒ KỸ THUẬT
    # ==========================================
    st.markdown("### 📈 Biểu đồ Kỹ thuật")
    st.write(
        "Sự thay đổi của đường Accuracy và Loss giúp củng cố nhận định mô hình không bị Overfitting nghiêm trọng. "
        "Confusion Matrix phản ánh hiệu suất thực tế trên từng nhãn nhận diện."
    )

    c1, c2 = st.columns(2)
    with c1:
        # Ưu tiên file ảnh nếu có, fallback về matplotlib
        cm_img_path = BASE_DIR / "anh.png" / "confusion_matrix.png"
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="Ma trận nhầm lẫn (Confusion Matrix)", use_container_width=True)
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
                    ax.text(j, i, int(cm_df.values[i, j]), ha="center", va="center", fontweight="bold")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            plot_confusion_matrix_custom()

    with c2:
        acc_loss_path = BASE_DIR / "anh.png" / "accuracy_loss.png"
        if acc_loss_path.exists():
            st.image(str(acc_loss_path), caption="Biểu đồ Accuracy và Loss", use_container_width=True)
        elif history_df is not None and "epoch" in history_df.columns:
            if {"train_accuracy", "val_accuracy"}.intersection(history_df.columns):
                draw_history_chart(history_df, ["train_accuracy", "val_accuracy"], "Accuracy theo Epoch", "Accuracy")
            elif {"accuracy", "val_accuracy"}.issubset(history_df.columns):
                draw_history_chart(history_df, ["accuracy", "val_accuracy"], "Accuracy theo Epoch", "Accuracy")
        else:
            # Fallback: vẽ cả loss + acc trực tiếp từ số liệu thực
            plot_training_curves()

    # Loss chart (nếu có file history)
    if history_df is not None and "epoch" in history_df.columns:
        st.markdown("**Đồ thị Loss:**")
        if {"train_loss", "val_loss"}.intersection(history_df.columns):
            draw_history_chart(history_df, ["train_loss", "val_loss"], "Loss theo Epoch", "Loss")
        elif {"loss", "val_loss"}.issubset(history_df.columns):
            draw_history_chart(history_df, ["loss", "val_loss"], "Loss theo Epoch", "Loss")

    st.markdown("---")

    # ==========================================
    # EPOCH TABLE
    # ==========================================
    with st.expander("📋 Bảng chi tiết 14 Epochs huấn luyện", expanded=False):
        epochs_table = pd.DataFrame({
            "Epoch":      list(range(1, 15)),
            "Train Loss": [0.1134, 0.0258, 0.0176, 0.0154, 0.0113, 0.0090, 0.0084, 0.0057, 0.0065, 0.0067, 0.0792, 0.0246, 0.0151, 0.0148],
            "Val Loss":   [0.0801, 0.0626, 0.0576, 0.0564, 0.0526, 0.0537, 0.0484, 0.0607, 0.0449, 0.0304, 0.0573, 0.0534, 0.0580, 0.0649],
            "Train Acc":  [0.9660, 0.9948, 0.9962, 0.9960, 0.9967, 0.9972, 0.9972, 0.9988, 0.9986, 0.9981, 0.9712, 0.9920, 0.9960, 0.9948],
            "Val Acc":    [0.9732, 0.9777, 0.9777, 0.9762, 0.9777, 0.9777, 0.9792, 0.9777, 0.9807, 0.9821, 0.9792, 0.9792, 0.9792, 0.9762],
            "Ghi chú":    ["Khởi động", "", "", "", "", "", "", "", "", "✅ Best (98.21%)", "⚡ Fine-tune (Phân đoạn 2)",
                           "", "", "Final / Early Stopping"],
        })
        st.dataframe(epochs_table, use_container_width=True, hide_index=True)
        st.caption("Epoch 10 là mốc mấu chốt đạt Val Accuracy 98.21% và Val Loss thấp nhất (0.0304). Bước sang Epoch 11, mô hình được unfreeze/reset LR để tinh chỉnh.")

    # ==========================================
    # PHÂN TÍCH SAI SỐ
    # ==========================================
    st.markdown("---")
    st.markdown("### 💡 Phân tích sai số (Error Analysis) và Giải pháp")

    c_err, c_fix = st.columns(2)
    with c_err:
        st.error("**❌ Trường hợp mô hình dự đoán sai (26 FN)**")
        st.markdown("""
📍 **Nguyên nhân gây ra 26 ca dự đoán nhầm 'Nhắm' thành 'Mở':**

- 💡 **Mắt sụp mí/Nhắm không hoàn toàn**: Mô hình nhận dạng là "Mở" thay vì "Nhắm", gây ra lỗi False Negative.
- 📐 **Góc xoay đầu hoặc khuất**: Việc Face Mesh trích xuất mắt không chính xác làm mô hình CNN bị rối.
- 👓 **Cản trở vật lý**: Kính ánh chói làm mất đường nét mi mắt khép kín.
""")

    with c_fix:
        st.success("**✅ Hướng cải thiện hệ thống**")
        st.markdown("""
🚀 **Giải pháp kỹ thuật:**

- 📊 **PERCLOS Metric**: Tính % thời gian mắt nhắm trong cửa sổ 60 giây (chuẩn ISO 15007) — ổn định hơn ngưỡng tĩnh hiện tại.
- 🔄 **Data Augmentation**: Random brightness ±40%, Gaussian blur, horizontal flip, Cutout để tăng khả năng generalize với điều kiện ánh sáng đêm.
- 🧠 **Temporal Modeling (LSTM/GRU)**: Học pattern chuỗi frame thay vì phân loại từng frame độc lập → giảm FP.
- ⚖️ **Class Weighting**: Gán trọng số cao hơn cho lớp Open_Eyes để cải thiện F1, xử lý imbalance tập val/test.
- 🎯 **Head Pose Estimation**: Thêm pitch/yaw/roll để lọc trường hợp EAR thấp do góc nghiêng đầu, không phải buồn ngủ.
""")

    st.markdown("---")
    st.markdown("### 📌 Tổng kết hiệu năng hệ thống")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy",   "95.25%", help="547 test samples")
    col2.metric("Precision (Closed)", "100.0%", delta="Zero FP", help="FP = 0")
    col3.metric("Recall (Closed)", "92.80%", help="FN = 26")
    col4.metric("Model Params",    "~2.3M",  help="MobileNetV2")


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


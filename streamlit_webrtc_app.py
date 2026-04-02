import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

from src.face_mesh_utils import FaceMeshDetector
from src.ear import compute_ear
from src.inference import EyeStateClassifier

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("😴 Drowsiness Detection (Webcam)")

# ===== Cấu hình =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "eye_state_model_final.keras"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map.json"

EAR_THRESHOLD = 0.17
MODEL_CLOSED_CONF_THRESHOLD = 0.7
DROWSY_FRAMES_THRESHOLD = 15
OPEN_EYES_FRAMES_THRESHOLD = 20

# ===== Cấu hình WebRTC =====
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===== Khởi tạo detector & classifier =====
@st.cache_resource
def load_models():
    detector = FaceMeshDetector()
    classifier = EyeStateClassifier(MODEL_PATH, LABEL_MAP_PATH)
    return detector, classifier

detector, classifier = load_models()

# ===== State management =====
if "drowsy_counter" not in st.session_state:
    st.session_state.drowsy_counter = 0
if "open_eye_counter" not in st.session_state:
    st.session_state.open_eye_counter = 0
if "alarm_on" not in st.session_state:
    st.session_state.alarm_on = False


def classify_eye_state(classifier, eye_crop):
    if eye_crop is None or eye_crop.size == 0:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "probs": {},
        }
    return classifier.predict(eye_crop)


def get_closed_prob(pred, label_name="Closed_Eyes"):
    probs = pred.get("probs", {})
    return float(probs.get(label_name, 0.0))


def is_eye_closed(pred, label_name="Closed_Eyes", threshold=MODEL_CLOSED_CONF_THRESHOLD):
    closed_prob = get_closed_prob(pred, label_name)
    return closed_prob >= threshold


class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = detector.process(img, draw=True)
        display = result.frame_bgr.copy()

        if not result.face_found:
            st.session_state.open_eye_counter = 0
            st.session_state.drowsy_counter = 0
            cv2.putText(display, "Face: Not detected", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return display

        # ===== Tính EAR =====
        left_ear = compute_ear(result.left_eye_points) if result.left_eye_points else 0.0
        right_ear = compute_ear(result.right_eye_points) if result.right_eye_points else 0.0
        avg_ear = (left_ear + right_ear) / 2.0

        # ===== Dự đoán model =====
        left_pred = classify_eye_state(classifier, result.left_eye_crop)
        right_pred = classify_eye_state(classifier, result.right_eye_crop)

        left_closed_by_model = is_eye_closed(left_pred)
        right_closed_by_model = is_eye_closed(right_pred)
        closed_by_model = left_closed_by_model and right_closed_by_model

        closed_by_ear = avg_ear < EAR_THRESHOLD
        eyes_closed = closed_by_ear and closed_by_model

        # ===== Logic cảnh báo =====
        if eyes_closed:
            st.session_state.drowsy_counter += 1
            st.session_state.open_eye_counter = 0
        else:
            st.session_state.drowsy_counter = 0
            st.session_state.open_eye_counter += 1

            if st.session_state.open_eye_counter >= OPEN_EYES_FRAMES_THRESHOLD:
                st.session_state.alarm_on = False

        if st.session_state.drowsy_counter >= DROWSY_FRAMES_THRESHOLD:
            st.session_state.alarm_on = True

        # ===== Vẽ cảnh báo =====
        if st.session_state.alarm_on:
            cv2.putText(display, "DROWSY ALERT!", (20, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # ===== Hiển thị thông tin =====
        h, w, _ = display.shape
        
        # Status
        if st.session_state.alarm_on:
            status_text = "DROWSY ALERT!"
            color = (0, 0, 255)
        else:
            status_text = "AWAKE"
            color = (0, 255, 0)

        cv2.putText(display, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # EAR
        cv2.putText(display, f"EAR: {avg_ear:.2f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Counter
        cv2.putText(display, f"Drowsy Frames: {st.session_state.drowsy_counter}/{DROWSY_FRAMES_THRESHOLD}",
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return display


webrtc_streamer(
    key="drowsiness",
    video_transformer_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

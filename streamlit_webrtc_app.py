import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

from src.face_mesh_utils import FaceMeshDetector
from src.ear import compute_ear

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("😴 Drowsiness Detection (Webcam)")

# ===== Cấu hình WebRTC (quan trọng để deploy) =====
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

detector = FaceMeshDetector()

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        result = detector.process(img, draw=True)

        if result.face_found:
            left_ear = compute_ear(result.left_eye_points)
            right_ear = compute_ear(result.right_eye_points)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < 0.18:
                cv2.putText(img, "DROWSY ALERT!", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "AWAKE", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.putText(img, f"EAR: {avg_ear:.2f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return img


webrtc_streamer(
    key="drowsiness",
    video_transformer_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

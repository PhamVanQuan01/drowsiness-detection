from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np


from src.alarm import AlarmPlayer
from src.ear import compute_ear
from src.face_mesh_utils import FaceMeshDetector
from src.inference import EyeStateClassifier


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "eye_state_model_final.keras"
LABEL_MAP_PATH = BASE_DIR / "models" / "label_map.json"
ALARM_PATH = BASE_DIR / "assets" / "alarm.wav"

# ===== Cấu hình =====
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# Ngưỡng EAR: càng thấp thì càng dễ coi là nhắm mắt
EAR_THRESHOLD = 0.17

# Xác suất model dự đoán Closed_Eyes để coi là mắt nhắm (chỉ dùng cho display)
MODEL_CLOSED_CONF_THRESHOLD = 0.75

# Số frame nhắm mắt liên tục để bật cảnh báo
DROWSY_FRAMES_THRESHOLD = 15

# Số frame mở mắt liên tục để tắt cảnh báo
OPEN_EYES_FRAMES_THRESHOLD = 20

# Hiển thị cửa sổ crop mắt
SHOW_EYE_WINDOWS = True


def classify_eye_state(classifier: EyeStateClassifier, eye_crop: np.ndarray | None) -> dict:
    if eye_crop is None or eye_crop.size == 0:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "probs": {},
        }
    return classifier.predict(eye_crop)


def get_closed_prob(pred: dict, label_name: str = "Closed_Eyes") -> float:
    probs = pred.get("probs", {})
    return float(probs.get(label_name, 0.0))


def is_eye_closed(pred: dict, label_name: str = "Closed_Eyes", threshold: float = MODEL_CLOSED_CONF_THRESHOLD) -> bool:
    closed_prob = get_closed_prob(pred, label_name)
    return closed_prob >= threshold


def put_status_text(frame: np.ndarray, lines: list[str], start_xy: tuple[int, int] = (20, 30)) -> None:
    x, y = start_xy
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28


def stop_alarm_if_needed(alarm: AlarmPlayer, alarm_on: bool) -> bool:
    if alarm_on:
        alarm.stop()
    return False


def main() -> None:
    print("[INFO] Đang khởi tạo model...")
    classifier = EyeStateClassifier(MODEL_PATH, LABEL_MAP_PATH)

    print("[INFO] Đang khởi tạo Face Mesh...")
    detector = FaceMeshDetector(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("[INFO] Đang khởi tạo âm thanh cảnh báo...")
    alarm = AlarmPlayer(ALARM_PATH)

    print("[INFO] Đang mở webcam...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        detector.close()
        alarm.close()
        raise RuntimeError("Không mở được webcam. Hãy kiểm tra camera hoặc đổi CAMERA_INDEX.")

    drowsy_counter = 0
    open_eye_counter = 0
    no_face_counter = 0
    alarm_on = False
    
    # ===== FPS Counter =====
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Không đọc được frame từ webcam.")
                break

            # ===== FPS Counter =====
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = time.time()

            frame = cv2.flip(frame, 1)
            result = detector.process(frame, draw=True)
            display = result.frame_bgr.copy()

            # ===== Không phát hiện được mặt =====
            if not result.face_found:
                no_face_counter += 1
                open_eye_counter = 0
                drowsy_counter = 0
                alarm_on = stop_alarm_if_needed(alarm, alarm_on)

                put_status_text(
                    display,
                    [
                        "Face: Not detected",
                        f"No-face frames: {no_face_counter}",
                        "Press 'q' to quit",
                    ],
                )

                cv2.imshow("Drowsiness Detection", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            no_face_counter = 0

            # ===== Tính EAR =====
            left_ear = compute_ear(result.left_eye_points) if result.left_eye_points else 0.0
            right_ear = compute_ear(result.right_eye_points) if result.right_eye_points else 0.0
            avg_ear = (left_ear + right_ear) / 2.0

            # ===== Dự đoán model =====
            left_pred = classify_eye_state(classifier, result.left_eye_crop)
            right_pred = classify_eye_state(classifier, result.right_eye_crop)

            left_closed_prob = get_closed_prob(left_pred)
            right_closed_prob = get_closed_prob(right_pred)

            left_closed_by_model = is_eye_closed(left_pred)
            right_closed_by_model = is_eye_closed(right_pred)

            # ===== LOGIC: Chỉ dùng EAR (đơn giản & reliable) =====
            # EAR thấp hơn ngưỡng thì coi là nhắm
            eyes_closed = avg_ear < EAR_THRESHOLD

            # ===== Logic bật/tắt cảnh báo =====
            if eyes_closed:
                drowsy_counter += 1
                open_eye_counter = 0
            else:
                 # MỞ MẮT → reset ngay
                drowsy_counter = 0
                open_eye_counter += 1

                # Khi mở mắt liên tục đủ số frame thì reset cảnh báo
                if open_eye_counter >= OPEN_EYES_FRAMES_THRESHOLD:
                    drowsy_counter = 0
                    alarm_on = stop_alarm_if_needed(alarm, alarm_on)

            # Bật cảnh báo nếu nhắm mắt liên tục đủ lâu
            if drowsy_counter >= DROWSY_FRAMES_THRESHOLD:
                if not alarm_on:
                    alarm.play()
                    alarm_on = True

            # ===== Vẽ cảnh báo =====
            if alarm_on:
                if int(time.time() * 2) % 2 == 0:
                    cv2.putText(
                        display,
                        "DROWSY ALERT!",
                        (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )

            # ===== Hiển thị thông tin =====
            # ===== UI HIỆN ĐẠI =====
            h, w, _ = display.shape

            # ===== Overlay mờ =====
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, int(h * 0.15)), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)

            # ===== STATUS =====
            if alarm_on:
                status_text = "DROWSY ALERT!"
                color = (0, 0, 255)
            else:
                status_text = "AWAKE"
                color = (0, 255, 0)

            # Scale theo màn hình
            font_scale = w / 900
            thickness = int(font_scale * 3)

            cv2.putText(
                display,
                status_text,
                (int(w * 0.03), int(h * 0.08)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            # ===== THANH PROGRESS =====
            bar_x1 = int(w * 0.03)
            bar_y1 = int(h * 0.12)
            bar_x2 = int(w * 0.35)
            bar_y2 = int(h * 0.15)

            # khung
            cv2.rectangle(display, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 2)

            # tính % buồn ngủ
            ratio = min(drowsy_counter / DROWSY_FRAMES_THRESHOLD, 1.0)

            # màu chuyển dần xanh → đỏ
            bar_color = (
                0,
                int(255 * (1 - ratio)),
                int(255 * ratio),
            )

            fill_x = int(bar_x1 + (bar_x2 - bar_x1) * ratio)

            cv2.rectangle(display, (bar_x1, bar_y1), (fill_x, bar_y2), bar_color, -1)

            # ===== STATS CORNER (góc phải) =====
            right_x = int(w * 0.72)
            right_y_start = int(h * 0.10)
            line_height = int(font_scale * 20)
            
            # Determine eye states
            left_eye_state = "CLOSED" if left_ear < EAR_THRESHOLD else "OPEN"
            right_eye_state = "CLOSED" if right_ear < EAR_THRESHOLD else "OPEN"
            overall_state = "CLOSED" if eyes_closed else "OPEN"
            
            left_eye_color = (0, 0, 255) if left_ear < EAR_THRESHOLD else (0, 255, 0)
            right_eye_color = (0, 0, 255) if right_ear < EAR_THRESHOLD else (0, 255, 0)
            overall_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            
            stats = [
                ("===== SYSTEM INFO =====", (255, 255, 255)),
                (f"FPS: {fps:.1f}", (0, 255, 0)),
                ("", (255, 255, 255)),
                ("===== EAR VALUES =====", (255, 255, 255)),
                (f"L-EAR: {left_ear:.3f}", (255, 255, 255)),
                (f"R-EAR: {right_ear:.3f}", (255, 255, 255)),
                (f"AVG-EAR: {avg_ear:.3f}", (255, 255, 255)),
                ("", (255, 255, 255)),
                ("===== EYE STATE =====", (255, 255, 255)),
                (f"Left: {left_eye_state}", left_eye_color),
                (f"Right: {right_eye_state}", right_eye_color),
                (f"Overall: {overall_state}", overall_color),
                ("", (255, 255, 255)),
                ("===== MODEL PRED =====", (255, 255, 255)),
                (f"L-Closed: {left_closed_prob:.2f}", (255, 255, 255)),
                (f"R-Closed: {right_closed_prob:.2f}", (255, 255, 255)),
                ("", (255, 255, 255)),
                ("===== COUNTERS =====", (255, 255, 255)),
                (f"Drowsy: {drowsy_counter}/{DROWSY_FRAMES_THRESHOLD}", (255, 255, 255)),
                (f"Open: {open_eye_counter}/{OPEN_EYES_FRAMES_THRESHOLD}", (255, 255, 255)),
                (f"No Face: {no_face_counter}", (255, 255, 255)),
            ]
            
            for i, (stat, color) in enumerate(stats):
                y_pos = right_y_start + (i * line_height)
                if y_pos < h - 30:  # Ensure not beyond bottom
                    cv2.putText(
                        display,
                        stat,
                        (right_x, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 0.45,
                        color,
                        1 if stat.startswith("=") else 1,
                        cv2.LINE_AA,
                    )

            # ===== Hướng dẫn =====
            cv2.putText(
                display,
                "Press Q to Quit",
                (int(w * 0.03), h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # ===== Hiển thị crop mắt =====
            if SHOW_EYE_WINDOWS:
                if result.left_eye_crop is not None:
                    cv2.imshow("Left Eye Crop", result.left_eye_crop)
                if result.right_eye_crop is not None:
                    cv2.imshow("Right Eye Crop", result.right_eye_crop)

            cv2.imshow("Drowsiness Detection", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        alarm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

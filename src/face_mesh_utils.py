from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np
import mediapipe as mp

from src.preprocessing import safe_crop, expand_box


# ===== MediaPipe face mesh landmark indices =====
# 6 điểm cho mỗi mắt để tính EAR theo chuẩn phổ biến
LEFT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

# Dùng thêm tập điểm rộng hơn để crop vùng mắt ổn định hơn
LEFT_EYE_CROP_IDX = [33, 133, 160, 158, 153, 144, 163, 157, 173, 154, 155, 246]
RIGHT_EYE_CROP_IDX = [362, 263, 385, 387, 373, 380, 390, 249, 466, 374, 381, 382]


@dataclass
class FaceMeshResult:
    """
    Kết quả xử lý một frame bằng MediaPipe Face Mesh.
    """
    frame_bgr: np.ndarray
    face_found: bool
    face_bbox: tuple[int, int, int, int] | None

    left_eye_points: list[tuple[int, int]]
    right_eye_points: list[tuple[int, int]]

    left_eye_crop: np.ndarray | None
    right_eye_crop: np.ndarray | None

    raw_landmarks: list[tuple[int, int]] | None = None
    meta: dict[str, Any] | None = None


class FaceMeshDetector:
    """
    Bộ phát hiện khuôn mặt + landmark mắt bằng MediaPipe Face Mesh.

    Đầu vào:
    - frame BGR

    Đầu ra:
    - FaceMeshResult
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        eye_crop_padding_x: float = 0.18,
        eye_crop_padding_y: float = 0.30,
    ) -> None:
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.eye_crop_padding_x = eye_crop_padding_x
        self.eye_crop_padding_y = eye_crop_padding_y

        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self.max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Không thể khởi tạo MediaPipe FaceMesh. Lỗi: {exc}"
            ) from exc

    def close(self) -> None:
        """
        Giải phóng tài nguyên MediaPipe.
        """
        try:
            self.face_mesh.close()
        except Exception:
            pass

    @staticmethod
    def _normalized_to_pixel(
        landmark,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        """
        Chuyển landmark chuẩn hóa [0,1] của MediaPipe về pixel.
        """
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)

        x = max(0, min(image_width - 1, x))
        y = max(0, min(image_height - 1, y))
        return x, y

    def _extract_all_landmarks(
        self,
        face_landmarks,
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int]]:
        """
        Chuyển toàn bộ landmarks của 1 khuôn mặt thành list pixel points.
        """
        points: list[tuple[int, int]] = []
        for lm in face_landmarks.landmark:
            points.append(self._normalized_to_pixel(lm, image_width, image_height))
        return points

    @staticmethod
    def _points_from_indices(
        all_points: list[tuple[int, int]],
        indices: list[int],
    ) -> list[tuple[int, int]]:
        """
        Lấy các điểm tương ứng theo danh sách index.
        """
        pts: list[tuple[int, int]] = []
        for idx in indices:
            if 0 <= idx < len(all_points):
                pts.append(all_points[idx])
        return pts

    @staticmethod
    def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        """
        Tính bounding box từ list điểm.
        """
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def _crop_eye_region(
        self,
        frame_bgr: np.ndarray,
        eye_points: list[tuple[int, int]],
    ) -> np.ndarray | None:
        """
        Crop vùng mắt từ tập điểm mắt.
        """
        bbox = self._bbox_from_points(eye_points)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = expand_box(
            x1,
            y1,
            x2,
            y2,
            pad_x=self.eye_crop_padding_x,
            pad_y=self.eye_crop_padding_y,
        )

        return safe_crop(frame_bgr, x1, y1, x2, y2)

    @staticmethod
    def _face_bbox_from_all_landmarks(
        all_points: list[tuple[int, int]]
    ) -> tuple[int, int, int, int] | None:
        """
        Tính bbox toàn khuôn mặt từ toàn bộ landmark.
        """
        if not all_points:
            return None

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def process(
        self,
        frame_bgr: np.ndarray,
        draw: bool = False,
        draw_landmarks: bool = False,
        draw_face_bbox: bool = False,
        draw_eye_boxes: bool = False,
    ) -> FaceMeshResult:
        """
        Xử lý một frame BGR và trả về landmark/crop mắt.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Frame đầu vào dạng BGR
        draw : bool
            Nếu True, cho phép vẽ overlay lên frame_bgr copy
        draw_landmarks : bool
            Vẽ các điểm landmark mắt
        draw_face_bbox : bool
            Vẽ khung mặt
        draw_eye_boxes : bool
            Vẽ khung crop mắt

        Returns
        -------
        FaceMeshResult
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr rỗng.")

        output_frame = frame_bgr.copy()
        h, w = output_frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return FaceMeshResult(
                frame_bgr=output_frame,
                face_found=False,
                face_bbox=None,
                left_eye_points=[],
                right_eye_points=[],
                left_eye_crop=None,
                right_eye_crop=None,
                raw_landmarks=None,
                meta={"reason": "no_face_detected"},
            )

        # Chỉ lấy 1 mặt đầu tiên
        face_landmarks = results.multi_face_landmarks[0]
        all_points = self._extract_all_landmarks(face_landmarks, w, h)

        face_bbox = self._face_bbox_from_all_landmarks(all_points)

        left_eye_points = self._points_from_indices(all_points, LEFT_EYE_EAR_IDX)
        right_eye_points = self._points_from_indices(all_points, RIGHT_EYE_EAR_IDX)

        left_eye_crop_points = self._points_from_indices(all_points, LEFT_EYE_CROP_IDX)
        right_eye_crop_points = self._points_from_indices(all_points, RIGHT_EYE_CROP_IDX)

        left_eye_crop = self._crop_eye_region(output_frame, left_eye_crop_points)
        right_eye_crop = self._crop_eye_region(output_frame, right_eye_crop_points)

        if draw:
            if draw_face_bbox and face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

            if draw_landmarks:
                for x, y in left_eye_points:
                    cv2.circle(output_frame, (x, y), 2, (0, 255, 0), -1)
                for x, y in right_eye_points:
                    cv2.circle(output_frame, (x, y), 2, (0, 255, 0), -1)

            if draw_eye_boxes:
                left_bbox = self._bbox_from_points(left_eye_crop_points)
                right_bbox = self._bbox_from_points(right_eye_crop_points)

                if left_bbox is not None:
                    lx1, ly1, lx2, ly2 = expand_box(
                        *left_bbox,
                        pad_x=self.eye_crop_padding_x,
                        pad_y=self.eye_crop_padding_y,
                    )
                    cv2.rectangle(output_frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)

                if right_bbox is not None:
                    rx1, ry1, rx2, ry2 = expand_box(
                        *right_bbox,
                        pad_x=self.eye_crop_padding_x,
                        pad_y=self.eye_crop_padding_y,
                    )
                    cv2.rectangle(output_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        return FaceMeshResult(
            frame_bgr=output_frame,
            face_found=True,
            face_bbox=face_bbox,
            left_eye_points=left_eye_points,
            right_eye_points=right_eye_points,
            left_eye_crop=left_eye_crop,
            right_eye_crop=right_eye_crop,
            raw_landmarks=all_points,
            meta={"reason": "ok"},
        )
    
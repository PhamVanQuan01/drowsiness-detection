from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_EAR_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDXS = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CROP_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CROP_IDXS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


@dataclass
class FaceMeshResult:
    face_found: bool
    frame_bgr: np.ndarray
    left_eye_points: list[tuple[int, int]] | None = None
    right_eye_points: list[tuple[int, int]] | None = None
    left_eye_crop: np.ndarray | None = None
    right_eye_crop: np.ndarray | None = None
    landmarks: Any = None


class FaceMeshDetector:
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _normalized_to_pixel(landmark, image_w: int, image_h: int) -> tuple[int, int]:
        x = min(max(int(landmark.x * image_w), 0), image_w - 1)
        y = min(max(int(landmark.y * image_h), 0), image_h - 1)
        return x, y

    def _extract_points(self, face_landmarks, image_w: int, image_h: int, indices: list[int]) -> list[tuple[int, int]]:
        pts = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            pts.append(self._normalized_to_pixel(lm, image_w, image_h))
        return pts

    @staticmethod
    def _crop_eye_region(frame_bgr: np.ndarray, points: list[tuple[int, int]], margin_ratio: float = 0.25) -> np.ndarray | None:
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            return None

        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)

        x1 = max(x_min - mx, 0)
        y1 = max(y_min - my, 0)
        x2 = min(x_max + mx, frame_bgr.shape[1] - 1)
        y2 = min(y_max + my, frame_bgr.shape[0] - 1)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2].copy()
        return crop

    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> FaceMeshResult:
        image_h, image_w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceMeshResult(face_found=False, frame_bgr=frame_bgr)

        output = frame_bgr.copy()
        face_landmarks = results.multi_face_landmarks[0]

        left_eye_points = self._extract_points(face_landmarks, image_w, image_h, LEFT_EYE_EAR_IDXS)
        right_eye_points = self._extract_points(face_landmarks, image_w, image_h, RIGHT_EYE_EAR_IDXS)

        left_eye_crop_points = self._extract_points(face_landmarks, image_w, image_h, LEFT_EYE_CROP_IDXS)
        right_eye_crop_points = self._extract_points(face_landmarks, image_w, image_h, RIGHT_EYE_CROP_IDXS)

        left_eye_crop = self._crop_eye_region(output, left_eye_crop_points)
        right_eye_crop = self._crop_eye_region(output, right_eye_crop_points)

        if draw:
            for x, y in left_eye_points + right_eye_points:
                cv2.circle(output, (x, y), 2, (0, 255, 0), -1)

            if left_eye_crop_points:
                lx = [p[0] for p in left_eye_crop_points]
                ly = [p[1] for p in left_eye_crop_points]
                cv2.rectangle(output, (min(lx), min(ly)), (max(lx), max(ly)), (255, 255, 0), 1)

            if right_eye_crop_points:
                rx = [p[0] for p in right_eye_crop_points]
                ry = [p[1] for p in right_eye_crop_points]
                cv2.rectangle(output, (min(rx), min(ry)), (max(rx), max(ry)), (255, 255, 0), 1)

        return FaceMeshResult(
            face_found=True,
            frame_bgr=output,
            left_eye_points=left_eye_points,
            right_eye_points=right_eye_points,
            left_eye_crop=left_eye_crop,
            right_eye_crop=right_eye_crop,
            landmarks=face_landmarks,
        )

    def close(self) -> None:
        self.face_mesh.close()

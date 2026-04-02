from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FaceMeshResult:
    face_found: bool
    frame_bgr: np.ndarray
    left_eye_points: list[tuple[int, int]] | None = None
    right_eye_points: list[tuple[int, int]] | None = None
    left_eye_crop: np.ndarray | None = None
    right_eye_crop: np.ndarray | None = None


class SimpleFaceDetector:
    """Simple face detector using Haar Cascades - no MediaPipe dependency"""
    
    def __init__(self):
        # Load Haar Cascade classifiers
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        # Cascades cho kính mắt
        self.glasses_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
    
    def _get_eye_contour_points(self, eye_region: np.ndarray, x_offset: int, y_offset: int) -> list[tuple[int, int]]:
        """Extract 6 points around eye region (similar to EAR index format)"""
        h, w = eye_region.shape[:2]
        points = [
            (x_offset, y_offset),
            (x_offset + w // 4, y_offset),
            (x_offset + w // 2, y_offset),
            (x_offset + w, y_offset + h // 2),
            (x_offset + w // 2, y_offset + h),
            (x_offset, y_offset + h // 2),
        ]
        return points
    
    @staticmethod
    def _crop_eye_region(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int, margin_ratio: float = 0.3) -> np.ndarray | None:
        """Crop eye region with margin"""
        mx = int(w * margin_ratio)
        my = int(h * margin_ratio)
        
        x1 = max(x - mx, 0)
        y1 = max(y - my, 0)
        x2 = min(x + w + mx, frame_bgr.shape[1])
        y2 = min(y + h + my, frame_bgr.shape[0])
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        crop = frame_bgr[y1:y2, x1:x2].copy()
        return crop if crop.size > 0 else None
    
    @staticmethod
    def _detect_pupils(face_gray: np.ndarray, face_x: int, face_y: int) -> np.ndarray:
        """Detect pupils as dark regions (threshholding)"""
        # Blur để giảm noise
        blurred = cv2.GaussianBlur(face_gray, (5, 5), 0)
        
        # Histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(blurred)
        
        # Threshold để tìm dark regions (pupils)
        _, thresh = cv2.threshold(equalized, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        pupils = []
        face_h, face_w = face_gray.shape
        
        # Tìm circular/elliptical contours (pupils thường tròn)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Kích thước pupil hợp lý
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Pupil phải gần hình tròn (aspect ratio gần 1)
                if 0.7 < aspect_ratio < 1.3:
                    pupils.append((x, y, w, h))
        
        # Return top 2 pupils (left và right)
        if len(pupils) >= 2:
            pupils = sorted(pupils, key=lambda p: p[0])  # Sort by x
            return np.array(pupils[:2])
        
        return np.array([])
    
    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> FaceMeshResult:
        """Process frame and detect face + eyes"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        output = frame_bgr.copy()
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        if len(faces) == 0:
            return FaceMeshResult(face_found=False, frame_bgr=output)
        
        # Get first (largest) face
        (x, y, w, h) = faces[0]
        
        # Draw face rectangle
        if draw:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Detect eyes in face region (cả bình thường và kính)
        face_gray = gray[y:y + h, x:x + w]
        
        # Thử detect eye thường
        eyes = self.eye_cascade.detectMultiScale(
            face_gray, 
            scaleFactor=1.05,  # Nhạy hơn
            minNeighbors=3,    # Giảm từ 4 xuống 3 để nhạy hơn
            minSize=(15, 15)   # Nhỏ hơn
        )
        
        # Nếu không tìm được eye thường, thử eye với kính
        if len(eyes) < 2:
            glasses = self.glasses_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(15, 15)
            )
            if len(glasses) >= 2:
                eyes = glasses
        
        # Nếu vẫn không tìm được, phát hiện dark regions (pupils)
        if len(eyes) < 2:
            eyes = self._detect_pupils(face_gray, x, y)
        
        left_eye_points = None
        left_eye_crop = None
        right_eye_points = None
        right_eye_crop = None
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate (left eye first)
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Left eye
            (ex1, ey1, ew1, eh1) = eyes[0]
            ex1 += x
            ey1 += y
            left_eye_points = self._get_eye_contour_points(face_gray[ey1-y:ey1-y+eh1, ex1-x:ex1-x+ew1], ex1, ey1)
            left_eye_crop = self._crop_eye_region(output, ex1, ey1, ew1, eh1)
            
            # Right eye
            (ex2, ey2, ew2, eh2) = eyes[1]
            ex2 += x
            ey2 += y
            right_eye_points = self._get_eye_contour_points(face_gray[ey2-y:ey2-y+eh2, ex2-x:ex2-x+ew2], ex2, ey2)
            right_eye_crop = self._crop_eye_region(output, ex2, ey2, ew2, eh2)
            
            # Draw eyes
            if draw:
                for i, eye in enumerate(eyes[:2]):
                    (ex, ey, ew, eh) = eye
                    cv2.rectangle(output, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
        
        return FaceMeshResult(
            face_found=True,
            frame_bgr=output,
            left_eye_points=left_eye_points,
            right_eye_points=right_eye_points,
            left_eye_crop=left_eye_crop,
            right_eye_crop=right_eye_crop,
        )
    
    def close(self):
        """Cleanup"""
        pass

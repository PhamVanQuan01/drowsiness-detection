from __future__ import annotations

import cv2
import numpy as np


def resize_with_padding(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize ảnh theo tỉ lệ gốc rồi thêm viền để đưa về kích thước target.
    Trả về ảnh BGR uint8.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def preprocess_eye_image(
    eye_bgr: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> np.ndarray:
    """
    Tiền xử lý ảnh mắt để đưa vào model MobileNetV2 đã train.
    - Input: ảnh BGR từ OpenCV
    - Output: tensor shape (1, H, W, 3)
    """
    padded = resize_with_padding(eye_bgr, target_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32)

    if normalize:
        x = x / 255.0

    x = np.expand_dims(x, axis=0)
    return x

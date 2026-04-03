from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def resize_frame_keep_aspect(
    frame: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """
    Resize ảnh nhưng giữ nguyên tỉ lệ.

    Chỉ nên truyền một trong hai:
    - target_width
    - target_height

    Parameters
    ----------
    frame : np.ndarray
        Ảnh đầu vào BGR
    target_width : int | None
    target_height : int | None

    Returns
    -------
    np.ndarray
        Ảnh đã resize
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame đầu vào rỗng.")

    h, w = frame.shape[:2]

    if target_width is None and target_height is None:
        return frame.copy()

    if target_width is not None and target_height is not None:
        # Nếu muốn resize đúng kích thước cố định
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    if target_width is not None:
        scale = target_width / float(w)
        new_h = max(1, int(h * scale))
        return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Chuyển BGR -> RGB
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Chuyển RGB -> BGR
    """
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def safe_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray | None:
    """
    Crop ảnh an toàn theo bounding box.

    Returns
    -------
    np.ndarray | None
        Ảnh crop hoặc None nếu box không hợp lệ
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def expand_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad_x: float = 0.15,
    pad_y: float = 0.25,
) -> tuple[int, int, int, int]:
    """
    Mở rộng bounding box theo tỉ lệ.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Bounding box gốc
    pad_x : float
        Tỉ lệ padding theo chiều ngang
    pad_y : float
        Tỉ lệ padding theo chiều dọc

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box mới
    """
    w = x2 - x1
    h = y2 - y1

    dx = int(w * pad_x)
    dy = int(h * pad_y)

    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize ảnh về kích thước cố định.

    Parameters
    ----------
    image : np.ndarray
    size : (width, height)

    Returns
    -------
    np.ndarray
    """
    if image is None or image.size == 0:
        raise ValueError("image đầu vào rỗng.")

    width, height = size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_image(
    image: np.ndarray,
    scale_01: bool = True,
) -> np.ndarray:
    """
    Chuẩn hóa ảnh cho model.

    Parameters
    ----------
    image : np.ndarray
    scale_01 : bool
        True -> chia 255 về [0,1]

    Returns
    -------
    np.ndarray
    """
    img = image.astype("float32")
    if scale_01:
        img /= 255.0
    return img


def prepare_eye_crop_for_model(
    eye_crop_bgr: np.ndarray,
    input_size: tuple[int, int] = (64, 64),
    convert_to_rgb: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Chuẩn bị crop mắt cho model.

    Quy trình:
    - resize
    - BGR -> RGB (nếu cần)
    - normalize
    - add batch dimension

    Parameters
    ----------
    eye_crop_bgr : np.ndarray
        Ảnh mắt crop từ frame BGR
    input_size : tuple[int, int]
        Kích thước model cần
    convert_to_rgb : bool
        Có đổi BGR sang RGB không
    normalize : bool
        Có chia 255 không

    Returns
    -------
    np.ndarray
        Tensor shape (1, H, W, C)
    """
    if eye_crop_bgr is None or eye_crop_bgr.size == 0:
        raise ValueError("eye_crop_bgr rỗng.")

    processed = resize_image(eye_crop_bgr, input_size)

    if convert_to_rgb:
        processed = bgr_to_rgb(processed)

    if normalize:
        processed = normalize_image(processed, scale_01=True)

    processed = np.expand_dims(processed, axis=0)
    return processed

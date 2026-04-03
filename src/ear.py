from __future__ import annotations

from typing import Iterable, Sequence
import math


PointLike = Sequence[float]


def _euclidean_distance(p1: PointLike, p2: PointLike) -> float:
    """
    Tính khoảng cách Euclid giữa 2 điểm 2D.

    Parameters
    ----------
    p1, p2 : Sequence[float]
        Điểm có dạng (x, y)

    Returns
    -------
    float
        Khoảng cách giữa 2 điểm
    """
    if len(p1) < 2 or len(p2) < 2:
        raise ValueError("Mỗi điểm phải có ít nhất 2 tọa độ (x, y).")

    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    return math.sqrt(dx * dx + dy * dy)


def validate_eye_points(eye_points: Iterable[PointLike]) -> list[PointLike]:
    """
    Kiểm tra và chuẩn hóa danh sách 6 điểm mắt.

    EAR tiêu chuẩn cần đúng 6 điểm:
        p1, p2, p3, p4, p5, p6

    Công thức:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    eye_points : Iterable[PointLike]

    Returns
    -------
    list[PointLike]
        Danh sách 6 điểm hợp lệ

    Raises
    ------
    ValueError
        Nếu số điểm không đúng hoặc dữ liệu không hợp lệ
    """
    pts = list(eye_points)

    if len(pts) != 6:
        raise ValueError(f"EAR cần đúng 6 điểm mắt, nhận được {len(pts)} điểm.")

    for idx, pt in enumerate(pts):
        if pt is None:
            raise ValueError(f"Điểm mắt tại vị trí {idx} là None.")
        if len(pt) < 2:
            raise ValueError(f"Điểm mắt tại vị trí {idx} không đủ tọa độ x, y.")

    return pts


def compute_ear(eye_points: Iterable[PointLike]) -> float:
    """
    Tính Eye Aspect Ratio (EAR) từ 6 điểm mắt.

    Thứ tự điểm:
        p1: khóe mắt trái
        p2: mí trên trái
        p3: mí trên phải
        p4: khóe mắt phải
        p5: mí dưới phải
        p6: mí dưới trái

    Công thức:
        EAR = (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))

    Giá trị EAR thấp -> mắt có xu hướng nhắm.

    Parameters
    ----------
    eye_points : Iterable[PointLike]

    Returns
    -------
    float
        Giá trị EAR. Nếu chiều ngang mắt quá nhỏ thì trả về 0.0
    """
    pts = validate_eye_points(eye_points)
    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = _euclidean_distance(p2, p6)
    vertical_2 = _euclidean_distance(p3, p5)
    horizontal = _euclidean_distance(p1, p4)

    if horizontal <= 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def compute_average_ear(
    left_eye_points: Iterable[PointLike] | None,
    right_eye_points: Iterable[PointLike] | None,
) -> tuple[float, float, float]:
    """
    Tính EAR mắt trái, mắt phải và EAR trung bình.

    Nếu một mắt lỗi/thiếu điểm thì mắt đó được tính là 0.0.

    Returns
    -------
    tuple[float, float, float]
        (left_ear, right_ear, avg_ear)
    """
    left_ear = 0.0
    right_ear = 0.0

    try:
        if left_eye_points:
            left_ear = compute_ear(left_eye_points)
    except Exception:
        left_ear = 0.0

    try:
        if right_eye_points:
            right_ear = compute_ear(right_eye_points)
    except Exception:
        right_ear = 0.0

    avg_ear = (left_ear + right_ear) / 2.0
    return float(left_ear), float(right_ear), float(avg_ear)


def ear_to_closed_score(
    ear: float,
    ear_open: float = 0.30,
    ear_closed: float = 0.18,
) -> float:
    """
    Chuyển EAR thành điểm 'mắt nhắm' trong khoảng [0, 1].

    Ý tưởng:
    - EAR cao gần mức mở mắt -> score gần 0
    - EAR thấp gần mức nhắm mắt -> score gần 1

    Parameters
    ----------
    ear : float
        Giá trị EAR hiện tại
    ear_open : float
        Mốc EAR xem như mắt mở rõ
    ear_closed : float
        Mốc EAR xem như mắt nhắm rõ

    Returns
    -------
    float
        Điểm closed score trong [0, 1]
    """
    if ear_open <= ear_closed:
        raise ValueError("ear_open phải lớn hơn ear_closed.")

    score = (ear_open - ear) / (ear_open - ear_closed)
    score = max(0.0, min(1.0, score))
    return float(score)

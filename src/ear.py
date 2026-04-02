from __future__ import annotations

from math import dist
from typing import Sequence


Point = tuple[float, float]


def compute_ear(eye_points: Sequence[Point]) -> float:
    """
    Tính EAR từ 6 điểm mắt theo công thức:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Thứ tự điểm:
    p1, p2, p3, p4, p5, p6
    """
    if len(eye_points) != 6:
        raise ValueError("EAR cần đúng 6 điểm landmark của mắt.")

    p1, p2, p3, p4, p5, p6 = eye_points

    horizontal = dist(p1, p4)
    if horizontal == 0:
        return 0.0

    vertical_1 = dist(p2, p6)
    vertical_2 = dist(p3, p5)

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)

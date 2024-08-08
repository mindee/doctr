import cv2
from typing import Sequence
import numpy as np
from typing import Tuple

__all__ = ['boundingRect', 'minAreaRect', 'fillPoly', 'morphologyEx']

def boundingRect(array: cv2.typing.MatLike) -> Sequence[int]:
    return cv2.boundingRect(array)

def minAreaRect(mat: cv2.typing.MatLike) -> Tuple[Sequence[float], Sequence[float], float]:
    return cv2.minAreaRect(mat)

def fillPoly(img: cv2.typing.MatLike, pts: Sequence[cv2.typing.MatLike], color: cv2.typing.Scalar) -> None:
    return cv2.fillPoly(img, pts, color)

def morphologyEx(src: np.ndarray, op: int, kernel: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(src, op, kernel)

# def boxPoints(box: cv2.typing.RotatedRect) -> np.ndarray:
#     return cv2.boxPoints(box)
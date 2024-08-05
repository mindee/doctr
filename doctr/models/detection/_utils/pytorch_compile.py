import torch
from torch import Tensor
import cv2
from typing import List, Sequence, Tuple
import numpy as np
import torch._dynamo.config

__all__ = [ 'boundingRect', 'minAreaRect', 'fillPoly', 'morphologyEx']

torch._dynamo.config.cache_size_limit = 30

def morphologyEx(src: np.ndarray, op: int, kernel: np.ndarray) -> np.ndarray:
    return _morphologyEx(torch.from_numpy(src), op, torch.from_numpy(kernel)).numpy()
# Register a custom_op for the morphologyEx
@torch.library.custom_op("cv2::morphologyEx", mutates_args=())
def _morphologyEx(src: torch.Tensor, op: int, kernel: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(cv2.morphologyEx(src.numpy(), op, kernel.numpy()))
# Register the FakeTensor as having the same size as the src
@_morphologyEx.register_fake
def _(src, op, kernel):
    return src

def boundingRect(array: cv2.typing.MatLike) -> Sequence[int]:
    return tuple(_boundingRect(Tensor(array)))

@torch.library.custom_op('cv2::boundingRect', mutates_args=())
def _boundingRect(array: Tensor) -> Tensor:
    return torch.LongTensor(cv2.boundingRect(array.numpy()))

@_boundingRect.register_fake
def _(array):
    return torch.empty((1, 4))

def minAreaRect(mat: cv2.typing.MatLike) -> Tuple[Sequence[float], Sequence[float], float]:
    packed = _minAreaRect(torch.from_numpy(mat))
    k = list(map(lambda x: x.numpy(), packed.split_with_sizes((2, 2, 1))))
    k[-1] = k[-1].item()
    return k

@torch.library.custom_op('cv2::minAreaRect', mutates_args=())
def _minAreaRect(mat: Tensor) -> Tensor:
    point, size, rot = cv2.minAreaRect(mat.numpy())
    return torch.FloatTensor([point[0], point[1], size[0], size[1], rot])

@_minAreaRect.register_fake
def _(mat):
    return torch.empty([5])

def fillPoly(img: cv2.typing.MatLike, pts: Sequence[cv2.typing.MatLike], color: cv2.typing.Scalar) -> None:
    _fillPoly(torch.from_numpy(img), torch.from_numpy(np.array(pts)), color)

@torch.library.custom_op('cv2::fillPoly', mutates_args=({'img'}))
def _fillPoly(img: Tensor, pts: Tensor, color: float) -> None:
    cv2.fillPoly(img.numpy(), [p.numpy() for p in pts], color)

# def boxPoints(box: cv2.typing.RotatedRect) -> cv2.typing.MatLike:
#     point, size, rot = box
#     return _boxPoints(torch.FloatTensor([point[0], point[1], size[0], size[1], rot])).numpy()

# @torch.library.custom_op('cv2::boxPoints', mutates_args=())
# def _boxPoints(box: Tensor) -> Tensor:
#     b = box.tolist()
#     return torch.from_numpy(cv2.boxPoints(((b[0], b[1]), (b[2], b[3]), b[4])))

# @_boxPoints.register_fake
# def _(box):
#     return torch.empty([4, 2])
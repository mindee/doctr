# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List
from .common_types import BoundingBox, Polygon4P

__all__ = ['bbox_to_polygon', 'polygon_to_bbox', 'resolve_enclosing_bbox']


def bbox_to_polygon(bbox: BoundingBox) -> Polygon4P:
    return (bbox[0], (bbox[1][0], bbox[0][1]), (bbox[0][0], bbox[1][1]), bbox[1])


def polygon_to_bbox(polygon: Polygon4P) -> BoundingBox:
    x, y = zip(*polygon)
    return ((min(x), min(y)), (max(x), max(y)))


def resolve_enclosing_bbox(bboxes: List[BoundingBox]) -> BoundingBox:
    x, y = zip(*[point for box in bboxes for point in box])
    return ((min(x), min(y)), (max(x), max(y)))

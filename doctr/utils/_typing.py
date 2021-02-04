# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple

__all__ = ['Point2D', 'BoundingBox', 'Polygon4P']

Point2D = Tuple[float, float]
BoundingBox = Tuple[Point2D, Point2D]
Polygon4P = Tuple[Point2D, Point2D, Point2D, Point2D]

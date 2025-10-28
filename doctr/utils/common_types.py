# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from pathlib import Path

__all__ = ["Point2D", "BoundingBox", "Polygon4P", "Polygon", "Bbox"]


Point2D = tuple[float, float]
BoundingBox = tuple[Point2D, Point2D]
Polygon4P = tuple[Point2D, Point2D, Point2D, Point2D]
Polygon = list[Point2D]
AbstractPath = str | Path
AbstractFile = AbstractPath | bytes
Bbox = tuple[float, float, float, float]

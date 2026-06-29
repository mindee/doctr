# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Point2D", "BoundingBox", "Polygon4P", "Polygon", "Bbox", "Sample"]


Point2D = tuple[float, float]
BoundingBox = tuple[Point2D, Point2D]
Polygon4P = tuple[Point2D, Point2D, Point2D, Point2D]
Polygon = list[Point2D]
AbstractPath = str | Path
AbstractFile = AbstractPath | bytes
Bbox = tuple[float, float, float, float]


@dataclass
class Sample:
    """Canonical data container for all transforms."""

    image: Any
    mask: Any | None = None
    target: np.ndarray | dict[str, np.ndarray] | None = None

    def replace(self, **kwargs) -> "Sample":
        return Sample(
            image=kwargs.get("image", self.image),
            mask=kwargs.get("mask", self.mask),
            target=kwargs.get("target", self.target),
        )

# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Tuple

from pydantic import BaseModel, Field


# Recognition output
class RecognitionOut(BaseModel):
    value: str = Field(..., example="Hello")


class DetectionOut(BaseModel):
    box: Tuple[float, float, float, float]


class OCROut(RecognitionOut, DetectionOut):
    pass

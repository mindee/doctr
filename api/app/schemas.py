# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Dict, List, Tuple, Union

from pydantic import BaseModel, Field


class RecognitionOut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    value: str = Field(..., examples=["Hello"])


class DetectionOut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    boxes: List[Tuple[float, float, float, float]]


class OCROut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    items: List[Dict[str, Union[str, Tuple[float, float, float, float]]]] = Field(
        ..., examples=[{"value": "example", "box": [0.0, 0.0, 0.0, 0.0]}]
    )


class KIEElement(BaseModel):
    class_name: str = Field(..., examples=["example"])
    items: List[Dict[str, Union[str, Tuple[float, float, float, float]]]] = Field(
        ..., examples=[{"value": "example", "box": [0.0, 0.0, 0.0, 0.0]}]
    )


class KIEOut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    predictions: List[KIEElement]

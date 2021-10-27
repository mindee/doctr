# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from fastapi import APIRouter, UploadFile, File
from typing import List

from doctr.io import decode_img_as_tensor
from app.vision import det_predictor
from app.schemas import DetectionOut


router = APIRouter()


@router.post("/", response_model=List[DetectionOut], status_code=200, summary="Perform text detection")
async def text_detection(file: UploadFile = File(...)):
    """Runs docTR text detection model to analyze the input"""
    img = decode_img_as_tensor(file.file.read())
    boxes, _ = det_predictor([img])[0]
    return [DetectionOut(box=box.tolist()) for box in boxes[:, :-1]]

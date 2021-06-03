# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from fastapi import APIRouter, UploadFile, File
from typing import List

from app.vision import decode_image, det_predictor
from app.schemas import DetectionOut


router = APIRouter()


@router.post("/", response_model=List[DetectionOut], status_code=200, summary="Perform text detection")
async def text_detection(file: UploadFile = File(...)):
    """Runs DocTR text detection model to analyze the input"""
    img = decode_image(file.file.read())
    [out, angle] = det_predictor([img], training=False)
    return [DetectionOut(box=box.tolist()) for box in out[0][:, :-1]]

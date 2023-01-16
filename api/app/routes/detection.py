# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from fastapi import APIRouter, File, UploadFile, status

from app.schemas import DetectionOut
from app.vision import det_predictor
from doctr.file_utils import CLASS_NAME
from doctr.io import decode_img_as_tensor

router = APIRouter()


@router.post("/", response_model=List[DetectionOut], status_code=status.HTTP_200_OK, summary="Perform text detection")
async def text_detection(file: UploadFile = File(...)):
    """Runs docTR text detection model to analyze the input image"""
    img = decode_img_as_tensor(file.file.read())
    boxes = det_predictor([img])[0]
    return [DetectionOut(box=box.tolist()) for box in boxes[CLASS_NAME][:, :-1]]

# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List

from app.schemas import OCROut
from app.vision import predictor
from fastapi import APIRouter, File, UploadFile, status

from doctr.io import decode_img_as_tensor

router = APIRouter()


@router.post("/", response_model=List[OCROut], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(file: UploadFile = File(...)):
    """Runs docTR OCR model to analyze the input image"""
    img = decode_img_as_tensor(file.file.read())
    out = predictor([img])

    return [OCROut(box=(*word.geometry[0], *word.geometry[1]), value=word.value)
            for word in out.pages[0].blocks[0].lines[0].words]

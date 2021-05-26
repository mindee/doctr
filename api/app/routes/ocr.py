# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from fastapi import APIRouter, UploadFile, File
from typing import List

from app.vision import decode_image, predictor
from app.schemas import OCROut


router = APIRouter()


@router.post("/", response_model=List[OCROut], status_code=200, summary="Perform OCR")
async def perform_ocr(file: UploadFile = File(...)):
    """Runs DocTR OCR model to analyze the input"""
    img = decode_image(file.file.read())
    out = predictor([img], training=False)

    return [OCROut(box=(*word.geometry[0], *word.geometry[1]), value=word.value)
            for word in out.pages[0].blocks[0].lines[0].words]

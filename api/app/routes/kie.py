# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Dict, List

from fastapi import APIRouter, File, UploadFile, status

from app.schemas import OCROut
from app.vision import kie_predictor
from doctr.io import decode_img_as_tensor

router = APIRouter()


@router.post("/", response_model=Dict[str, List[OCROut]], status_code=status.HTTP_200_OK, summary="Perform KIE")
async def perform_kie(file: UploadFile = File(...)):
    """Runs docTR KIE model to analyze the input image"""
    img = decode_img_as_tensor(file.file.read())
    out = kie_predictor([img])

    return {
        class_name: [
            OCROut(box=(*word.geometry[0], *word.geometry[1]), value=word.value)
            for block in out.pages[0].predictions[class_name]
            for line in block.lines
            for word in line.words
        ]
        for class_name in out.pages[0].predictions.keys()
    }
# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from fastapi import APIRouter, UploadFile, File

from doctr.io import decode_img_as_tensor
from app.vision import reco_predictor
from app.schemas import RecognitionOut


router = APIRouter()


@router.post("/", response_model=RecognitionOut, status_code=200, summary="Perform text recognition")
async def text_recognition(file: UploadFile = File(...)):
    """Runs DocTR text recognition model to analyze the input"""
    img = decode_image(file.file.read())
    out = reco_predictor([img], training=False)
    return RecognitionOut(value=out[0][0])

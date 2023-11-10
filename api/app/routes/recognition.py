# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from fastapi import APIRouter, File, UploadFile, status

from app.schemas import RecognitionOut
from app.vision import reco_predictor
from doctr.io import decode_img_as_tensor

router = APIRouter()


@router.post("/", response_model=RecognitionOut, status_code=status.HTTP_200_OK, summary="Perform text recognition")
async def text_recognition(file: UploadFile = File(...)):
    """Runs docTR text recognition model to analyze the input image"""
    img = decode_img_as_tensor(file.file.read())
    out = reco_predictor([img])
    return RecognitionOut(value=out[0][0])

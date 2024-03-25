# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import RecognitionOut
from app.vision import reco_predictor
from doctr.io import DocumentFile

router = APIRouter()


@router.post(
    "/", response_model=List[RecognitionOut], status_code=status.HTTP_200_OK, summary="Perform text recognition"
)
async def text_recognition(files: List[UploadFile] = [File(...)]):
    """Runs docTR text recognition model to analyze the input image"""
    words: List[RecognitionOut] = []
    for file in files:
        mime_type = file.content_type
        if mime_type in ["image/jpeg", "image/png"]:
            content = DocumentFile.from_images([await file.read()])
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file format for recognition endpoint: {mime_type}"
            )

        words.append(RecognitionOut(name=file.filename or "", value=reco_predictor(content)[0][0]))

    return words

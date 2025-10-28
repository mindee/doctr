# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas import RecognitionIn, RecognitionOut
from app.utils import get_documents
from app.vision import init_predictor

router = APIRouter()


@router.post(
    "/", response_model=list[RecognitionOut], status_code=status.HTTP_200_OK, summary="Perform text recognition"
)
async def text_recognition(request: RecognitionIn = Depends(), files: list[UploadFile] = [File(...)]):
    """Runs docTR text recognition model to analyze the input image"""
    try:
        predictor = init_predictor(request)
        content, filenames = await get_documents(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        RecognitionOut(name=filename, value=res[0], confidence=round(res[1], 2))
        for res, filename in zip(predictor(content), filenames)
    ]

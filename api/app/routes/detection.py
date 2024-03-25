# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import DetectionOut
from app.vision import det_predictor
from doctr.file_utils import CLASS_NAME
from doctr.io import DocumentFile

router = APIRouter()


@router.post("/", response_model=List[DetectionOut], status_code=status.HTTP_200_OK, summary="Perform text detection")
async def text_detection(files: List[UploadFile] = [File(...)]):
    """Runs docTR text detection model to analyze the input image"""
    boxes: List[DetectionOut] = []
    for file in files:
        mime_type = file.content_type
        if mime_type in ["image/jpeg", "image/png"]:
            content = DocumentFile.from_images([await file.read()])
        elif mime_type == "application/pdf":
            content = DocumentFile.from_pdf(await file.read())
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format for detection endpoint: {mime_type}")

        boxes.append(
            DetectionOut(
                name=file.filename or "", boxes=[box.tolist() for box in det_predictor(content)[0][CLASS_NAME][:, :-1]]
            )
        )

    return boxes

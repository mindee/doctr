# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import OCROut
from app.vision import predictor
from doctr.io import DocumentFile

router = APIRouter()


@router.post("/", response_model=List[OCROut], status_code=status.HTTP_200_OK, summary="Perform OCR")
async def perform_ocr(files: List[UploadFile] = [File(...)]):
    """Runs docTR OCR model to analyze the input image"""
    results: List[OCROut] = []
    for file in files:
        mime_type = file.content_type
        if mime_type in ["image/jpeg", "image/png"]:
            content = DocumentFile.from_images([await file.read()])
        elif mime_type == "application/pdf":
            content = DocumentFile.from_pdf(await file.read())
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format for OCR endpoint: {mime_type}")

        out = predictor(content)
        for page in out.pages:
            results.append(
                OCROut(
                    name=file.filename or "",
                    items=[
                        dict(value=word.value, box=(*word.geometry[0], *word.geometry[1]))
                        for block in page.blocks
                        for line in block.lines
                        for word in line.words
                    ],
                )
            )

    return results

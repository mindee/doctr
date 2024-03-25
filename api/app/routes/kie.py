# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas import KIEElement, KIEOut
from app.vision import kie_predictor
from doctr.io import DocumentFile

router = APIRouter()


@router.post("/", response_model=List[KIEOut], status_code=status.HTTP_200_OK, summary="Perform KIE")
async def perform_kie(files: List[UploadFile] = [File(...)]):
    """Runs docTR KIE model to analyze the input image"""
    results: List[KIEOut] = []
    for file in files:
        mime_type = file.content_type
        if mime_type in ["image/jpeg", "image/png"]:
            content = DocumentFile.from_images([await file.read()])
        elif mime_type == "application/pdf":
            content = DocumentFile.from_pdf(await file.read())
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format for KIE endpoint: {mime_type}")

        out = kie_predictor(content)

        for page in out.pages:
            results.append(
                KIEOut(
                    name=file.filename or "",
                    predictions=[
                        KIEElement(
                            class_name=class_name,
                            items=[
                                dict(value=prediction.value, box=(*prediction.geometry[0], *prediction.geometry[1]))
                                for prediction in page.predictions[class_name]
                            ],
                        )
                        for class_name in page.predictions.keys()
                    ],
                )
            )

    return results

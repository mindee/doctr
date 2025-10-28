# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas import DetectionIn, DetectionOut
from app.utils import get_documents, resolve_geometry
from app.vision import init_predictor
from doctr.file_utils import CLASS_NAME

router = APIRouter()


@router.post("/", response_model=list[DetectionOut], status_code=status.HTTP_200_OK, summary="Perform text detection")
async def text_detection(request: DetectionIn = Depends(), files: list[UploadFile] = [File(...)]):
    """Runs docTR text detection model to analyze the input image"""
    try:
        predictor = init_predictor(request)
        content, filenames = await get_documents(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return [
        DetectionOut(
            name=filename,
            geometries=[
                geom[:-1].tolist() if geom.shape == (5,) else resolve_geometry(geom[:4].tolist())
                for geom in doc[CLASS_NAME]
            ],
        )
        for doc, filename in zip(predictor(content), filenames)
    ]

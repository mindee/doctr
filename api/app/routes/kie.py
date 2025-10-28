# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas import KIEElement, KIEIn, KIEOut
from app.utils import get_documents, resolve_geometry
from app.vision import init_predictor

router = APIRouter()


@router.post("/", response_model=list[KIEOut], status_code=status.HTTP_200_OK, summary="Perform KIE")
async def perform_kie(request: KIEIn = Depends(), files: list[UploadFile] = [File(...)]):
    """Runs docTR KIE model to analyze the input image"""
    try:
        predictor = init_predictor(request)
        content, filenames = await get_documents(files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    out = predictor(content)

    results = [
        KIEOut(
            name=filenames[i],
            orientation=page.orientation,
            language=page.language,
            dimensions=page.dimensions,
            predictions=[
                KIEElement(
                    class_name=class_name,
                    items=[
                        dict(
                            value=prediction.value,
                            geometry=resolve_geometry(prediction.geometry),
                            objectness_score=round(prediction.objectness_score, 2),
                            confidence=round(prediction.confidence, 2),
                            crop_orientation=prediction.crop_orientation,
                        )
                        for prediction in page.predictions[class_name]
                    ],
                )
                for class_name in page.predictions.keys()
            ],
        )
        for i, page in enumerate(out.pages)
    ]

    return results

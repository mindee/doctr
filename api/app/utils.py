# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Any

import numpy as np
from fastapi import UploadFile

from doctr.io import DocumentFile


def resolve_geometry(
    geom: Any,
) -> tuple[float, float, float, float] | tuple[float, float, float, float, float, float, float, float]:
    if len(geom) == 4:
        return (*geom[0], *geom[1], *geom[2], *geom[3])
    return (*geom[0], *geom[1])


async def get_documents(files: list[UploadFile]) -> tuple[list[np.ndarray], list[str]]:  # pragma: no cover
    """Convert a list of UploadFile objects to lists of numpy arrays and their corresponding filenames

    Args:
        files: list of UploadFile objects

    Returns:
        tuple[list[np.ndarray], list[str]]: list of numpy arrays and their corresponding filenames

    """
    filenames = []
    docs = []
    for file in files:
        mime_type = file.content_type
        if mime_type in ["image/jpeg", "image/png"]:
            docs.extend(DocumentFile.from_images([await file.read()]))
            filenames.append(file.filename or "")
        elif mime_type == "application/pdf":
            pdf_content = DocumentFile.from_pdf(await file.read())
            docs.extend(pdf_content)
            filenames.extend([file.filename] * len(pdf_content) or [""] * len(pdf_content))
        else:
            raise ValueError(f"Unsupported file format: {mime_type} for file {file.filename}")

    return docs, filenames

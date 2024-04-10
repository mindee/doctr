# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Any, List, Tuple, Union

import numpy as np
from fastapi import UploadFile

from doctr.io import DocumentFile


def resolve_geometry(
    geom: Any,
) -> Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float, float, float, float]]:
    if len(geom) == 4:
        return (*geom[0], *geom[1], *geom[2], *geom[3])
    return (*geom[0], *geom[1])


async def get_documents(files: List[UploadFile]) -> Tuple[List[np.ndarray], List[str]]:  # pragma: no cover
    """Convert a list of UploadFile objects to lists of numpy arrays and their corresponding filenames

    Args:
    ----
        files: list of UploadFile objects

    Returns:
    -------
        Tuple[List[np.ndarray], List[str]]: list of numpy arrays and their corresponding filenames

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
            filenames.append(file.filename or "" * len(pdf_content))
        else:
            raise ValueError(f"Unsupported file format: {mime_type} for file {file.filename}")

    return docs, filenames

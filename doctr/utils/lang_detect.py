# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple

import fasttext
from scipy.stats import entropy

from doctr.utils import download_from_url

__all__ = ['detect_language']
URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def detect_language(text: str) -> Tuple[str, float]:
    """Get languages of a textl using fasttext model.
    Get the language with the highest probability or no language if only a few words or a high entropy
    Args:
        text (str): text
    Returns:
        The detected language in ISO 639 code and confidence score
    """
    archive_path = download_from_url(URL, cache_subdir='language_detection')
    print(archive_path)
    K = 8
    MAX_ENTROPY = 3
    TH_ENTROPY = 0.9
    FASTTEXT_MODEL = fasttext.load_model(str(archive_path))

    prediction = FASTTEXT_MODEL.predict(text.lower(), k=K)
    langs = prediction[0][0].replace("__label__", "")
    if (
        len(text) <= 1
        or (len(text) <= 5 and prediction[1][0] <= 0.2)
        or (entropy(prediction[1], base=2) / MAX_ENTROPY > TH_ENTROPY)
    ):
        return "unknown", 0.0
    return langs, prediction[1][0]

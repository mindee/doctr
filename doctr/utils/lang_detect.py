# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from os.path import dirname, join
from typing import Tuple

import fasttext
from scipy.stats import entropy

__all__ = ['detect_language']


def detect_language(text: str) -> Tuple[str, float]:
    """Get languages of a textl using fasttext model.
    Get the language with the highest probability or no language if only a few words or a high entropy
    Args:
        text (str): text
    Returns:
        The detected language in ISO 639 code and confidence score
    """
    K = 8
    MAX_ENTROPY = 3
    TH_ENTROPY = 0.9
    FASTTEXT_MODEL = fasttext.FastText._FastText(join(dirname(__file__), "../../lid.176.ftz"))

    prediction = FASTTEXT_MODEL.predict(text.lower(), k=K)
    langs = prediction[0][0].replace("__label__", "")
    if (
        len(text) <= 1
        or (len(text) <= 5 and prediction[1][0] <= 0.2)
        or (entropy(prediction[1], base=2) / MAX_ENTROPY > TH_ENTROPY)
    ):
        return "unknown", 0.0
    return langs, prediction[1][0]

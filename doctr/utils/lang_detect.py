# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple

from langdetect import detect_langs

__all__ = ['detect_language']


def detect_language(text: str) -> Tuple[str, float]:
    """Detects Language of text

    Args:
        text: text to detect language from

    Returns:
        The detected language and confidence score
    """

    if text == "":
        return "empty text", 1.0
    lang = detect_langs(text)[0]
    return lang.lang, lang.prob

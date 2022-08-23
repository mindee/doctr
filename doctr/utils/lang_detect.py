# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple

from lingua import Language, LanguageDetectorBuilder

from doctr.datasets.vocabs import LANGUAGES

__all__ = ['get_language']

languages = [Language._member_map_[item.upper()] for item in LANGUAGES if
             item.upper() in Language._member_map_.keys()]
MODEL = LanguageDetectorBuilder.from_languages(*languages).build()


def get_language(text: str) -> Tuple[str, float]:
    """Get languages of a text using lingua library.
    Get the language with the highest probability or no language if only a few words or no language was identified
    Args:
        text (str): text
    Returns:
        The detected language in ISO 639 code and confidence score
    """
    predictions = MODEL.compute_language_confidence_values(text.lower())
    if len(text) <= 5 or not predictions:
        return "unknown", 0.0
    lang, prob = predictions[0]
    return lang.iso_code_639_1.name.lower(), prob

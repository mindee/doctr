# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple

from lingua import Language, LanguageDetectorBuilder

from doctr.datasets.vocabs import VOCABS

__all__ = ['get_language']

potential_languages = [item.upper() for item in VOCABS.keys()]
language_dict = Language._member_map_
languages = []
for item in potential_languages:
    language = language_dict.get(item, None)
    if language:
        languages.append(language)


def get_language(text: str) -> Tuple[str, float]:
    """Get languages of a textl using fasttext model.
    Get the language with the highest probability or no language if only a few words or a high entropy
    Args:
        text (str): text
    Returns:
        The detected language in ISO 639 code and confidence score
    """

    model = LanguageDetectorBuilder.from_languages(*languages).build()
    predictions = model.compute_language_confidence_values(text.lower())
    if (
        len(text) <= 5
        or not predictions
    ):
        return "unknown", 0.0
    lang, prob = predictions[0]
    return lang.iso_code_639_1.name.lower(), prob

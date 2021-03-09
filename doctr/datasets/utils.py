# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import string
import unicodedata
import numpy as np
from typing import List

from .vocabs import VOCABS

__all__ = ['translate', 'encode_sequence', 'decode_sequence']


def translate(
    input_string: str,
    vocab_name: str,
    unknown_char: str = '■',
) -> str:
    """Translate a string input in a given vocabulary

    Args:
        input_string: input string to translate
        vocab_name: vocabulary to use (french, latin, ...)
        unknown_char: unknown character for non-translatable characters

    Returns:
        A string translated in a given vocab"""

    if VOCABS.get(vocab_name) is None:
        raise KeyError("output vocabulary must be in vocabs dictionnary")

    translated = ''
    for char in input_string:
        if char not in VOCABS[vocab_name]:
            # we need to translate char into a vocab char
            if char in string.whitespace:
                # remove whitespaces
                continue
            # normalize character if it is not in vocab
            char = unicodedata.normalize('NFD', char).encode('ascii', 'ignore').decode('ascii')
            if char == '' or char not in VOCABS[vocab_name]:
                # if normalization fails or char still not in vocab, return unknown character)
                char = unknown_char
        translated += char
    return translated


def encode_sequence(
    input_string: str,
    mapping: str,
) -> List[str]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
        input_string: string to encode
        mapping: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A list encoding the input_string"""

    encoded = list(map(mapping.index, input_string))
    return encoded


def decode_sequence(
    input_array: np.array,
    mapping: str,
) -> str:
    """Given a predefined mapping, decode the sequence of numbers to a string

    Args:
        input_array: array to decode
        mapping: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A string, decoded from input_array"""

    if not input_array.dtype == np.int_ or input_array.max() >= len(mapping):
        raise AssertionError("Input must be an array of int, with max less than mapping size")
    decoded = ''.join(mapping[idx] for idx in input_array)
    return decoded

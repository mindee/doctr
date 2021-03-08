# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import string
import unicodedata
import numpy as np
from typing import Dict

__all__ = ['translate', 'encode_sequence', 'decode_sequence']


vocabs: Dict[str, str] = {
    'digits': string.digits,
    'ascii_letters': string.ascii_letters,
    'punctuation': string.punctuation,
    'currency': '£€¥¢฿',
    'latin': string.digits + string.ascii_letters + string.punctuation + '°',
    'french': string.digits + string.ascii_letters + string.punctuation + '°' + 'àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ' + '£€¥¢฿',
    'tfrecords_training': '3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î'
                          '£|za1ù8,OG€P-kçHëÀÂ2É/ûIJ\'j(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l'
}


def translate(
    input_string: str,
    vocab: str,
) -> str:
    """Translate a string input in a given vocabulary

    Args:
        input_string: input string to translate
        vocab: vocabulary to use (french, latin, ...)

    Returns:
        A string translated in a given vocab"""

    if vocab not in vocabs.keys():
        raise AttributeError("output vocabulary must be in vocabs dictionnary")

    translated = ''
    for char in input_string:
        if char not in vocabs[vocab]:
            # we need to translate char into a vocab char
            if char in string.whitespace:
                # remove whitespaces
                continue
            # normalize character if it is not in vocab
            char = unicodedata.normalize('NFD', char).encode('ascii', 'ignore').decode('ascii')
            if char == '' or char not in vocabs[vocab]:
                # if normalization fails or char still not in vocab, return a black square (unknown symbol)
                char = '■'
        translated += char
    return translated


def encode_sequence(
    input_string: str,
    mapping: str,
) -> np.array:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
        input_string: string to encode
        mapping: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A tensor encoding the input_string"""

    encoded = []
    for char in input_string:
        if char not in mapping:
            raise AttributeError("Input string vocabulary does not match encoding vocabulary")
        encoded.append(mapping.index(char))
    return np.array(encoded)


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

    if not input_array.dtype == np.int_ or max(input_array) >= len(mapping):
        raise AttributeError("Input must be an array of int, with max less than mapping size")
    decoded = ''
    for index in list(input_array):
        decoded += mapping[index]
    return decoded

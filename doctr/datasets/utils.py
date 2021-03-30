# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import string
import unicodedata
import numpy as np
from typing import List, Optional, Any

from .vocabs import VOCABS

__all__ = ['translate', 'encode_sequence', 'decode_sequence', 'encode_sequences']


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
    vocab: str,
) -> List[str]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
        input_string: string to encode
        vocab: vocabulary (string), the encoding is given by the indexing of the character sequence

    Returns:
        A list encoding the input_string"""

    return list(map(vocab.index, input_string))


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


def encode_sequences(
    sequences: List[str],
    vocab: str,
    target_size: Optional[int] = None,
    eos: int = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Encode character sequences using a given vocab as mapping

    Args:
        sequences: the list of character sequences of size N
        vocab: the ordered vocab to use for encoding
        target_size: maximum length of the encoded data
        eos: encoding of End Of String

    Returns:
        the padded encoded data as a tensor
    """

    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")

    if not isinstance(target_size, int):
        target_size = max(len(w) for w in sequences)

    # Pad all sequences
    encoded_data = np.full([len(sequences), target_size], eos, dtype=np.int32)

    for idx, seq in enumerate(sequences):
        encoded_seq = encode_sequence(seq, vocab)
        encoded_data[idx, :min(len(encoded_seq), target_size)] = encoded_seq[:min(len(encoded_seq), target_size)]

    return encoded_data

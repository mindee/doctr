# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import re
import string
import unicodedata
from collections.abc import Sequence
from collections.abc import Sequence as SequenceType
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from PIL import Image

from doctr.io.image import get_img_shape
from doctr.utils.geometry import convert_to_relative_coords, extract_crops, extract_rcrops

from .vocabs import VOCABS

__all__ = [
    "translate",
    "tokenize_latex",
    "normalize_latex",
    "encode_string",
    "decode_sequence",
    "encode_sequences",
    "pre_transform_multiclass",
    "crop_bboxes_from_image",
    "convert_target_to_relative",
]

ImageTensor = TypeVar("ImageTensor")


def translate(
    input_string: str,
    vocab_name: str,
    unknown_char: str = "■",
) -> str:
    """Translate a string input in a given vocabulary

    Args:
        input_string: input string to translate
        vocab_name: vocabulary to use (french, latin, ...)
        unknown_char: unknown character for non-translatable characters

    Returns:
        A string translated in a given vocab
    """
    if VOCABS.get(vocab_name) is None:
        raise KeyError("output vocabulary must be in vocabs dictionary")

    translated = ""
    for char in input_string:
        if char not in VOCABS[vocab_name]:
            # we need to translate char into a vocab char
            if char in string.whitespace:
                # remove whitespaces
                continue
            # normalize character if it is not in vocab
            char = unicodedata.normalize("NFD", char).encode("ascii", "ignore").decode("ascii")
            if char == "" or char not in VOCABS[vocab_name]:
                # if normalization fails or char still not in vocab, return unknown character)
                char = unknown_char
        translated += char
    return translated


# A LaTeX token: an environment delimiter (e.g. \begin{array}), a command (e.g. \frac, \operatorname*), a control
# symbol (e.g. \{, \, or \\) or any other non-whitespace character
_LATEX_TOKEN = re.compile(r"\\(?:begin|end)\s*\{[a-zA-Z]+\*?\}|\\[a-zA-Z]+\*?|\\.|\S", re.DOTALL)


def tokenize_latex(latex: str) -> list[str]:
    r"""Split a LaTeX string into tokens: commands, environment delimiters, control symbols and single characters

    >>> from doctr.datasets import tokenize_latex
    >>> tokenize_latex(r"\frac{a}{b^2}")
    ['\\frac', '{', 'a', '}', '{', 'b', '^', '2', '}']

    The whitespaces are not part of the tokens (a control space `\ ` is tokenized as `\space`, which renders the same):
    tokenizing a string and joining the tokens with spaces gives a normalized string which tokenizes into the same
    tokens.

    Args:
        latex: the LaTeX string to tokenize

    Returns:
        the list of tokens
    """
    tokens = []
    for tok in _LATEX_TOKEN.findall(latex):
        if tok.startswith(("\\begin", "\\end")):
            # Drop the whitespaces between an environment delimiter and its name (e.g. "\begin {array}")
            tok = re.sub(r"\s+", "", tok)
        elif len(tok) == 2 and tok[1].isspace():
            tok = "\\space"
        tokens.append(tok)
    return tokens


def normalize_latex(latex: str) -> str:
    r"""Normalize a LaTeX string into space-separated tokens (cf. :func:`tokenize_latex`)

    >>> from doctr.datasets import normalize_latex
    >>> normalize_latex(r"\frac{a}{b^2}")
    '\\frac { a } { b ^ 2 }'

    Args:
        latex: the LaTeX string to normalize

    Returns:
        the normalized string, rendering the same as the input
    """
    return " ".join(tokenize_latex(latex))


@lru_cache(maxsize=32)
def _vocab_lookup_table(vocab: str | tuple[str, ...]) -> dict[str, int]:
    """Cached token -> index table for a vocab (the first occurrence of a duplicated token wins)"""
    return {token: idx for idx, token in reversed(list(enumerate(vocab)))}


def _split_tokens(input_string: str, vocab: str | Sequence[str]) -> list[str]:
    """Split a string into the units of a vocab: characters for a string vocab, whitespace-separated tokens for a
    list of tokens"""
    return list(input_string) if isinstance(vocab, str) else input_string.split()


def encode_string(
    input_string: str,
    vocab: str | Sequence[str],
) -> list[int]:
    """Given a predefined mapping, encode the string to a sequence of numbers

    Args:
        input_string: string to encode
        vocab: vocabulary, the encoding is given by the indexing of its units: a string of characters, or a list of
            tokens (e.g. LaTeX commands), in which case the input string is split on whitespaces

    Returns:
        A list encoding the input_string
    """
    table = _vocab_lookup_table(vocab if isinstance(vocab, str) else tuple(vocab))
    tokens = _split_tokens(input_string, vocab)
    try:
        return [table[token] for token in tokens]
    except KeyError as e:
        missing = [token for token in tokens if token not in table]
        raise ValueError(
            f"Some characters cannot be found in 'vocab': {set(missing)}.\n"
            f"Please check the input string `{input_string}` and the vocabulary `{vocab}`"
        ) from e


def decode_sequence(
    input_seq: np.ndarray | SequenceType[int],
    mapping: str | Sequence[str],
) -> str:
    """Given a predefined mapping, decode the sequence of numbers to a string

    Args:
        input_seq: array to decode
        mapping: vocabulary, the encoding is given by the indexing of its units: a string of characters, or a list of
            tokens, in which case the decoded tokens are joined with spaces

    Returns:
        A string, decoded from input_seq
    """
    if not isinstance(input_seq, (Sequence, np.ndarray)):
        raise TypeError("Invalid sequence type")
    if isinstance(input_seq, np.ndarray) and not np.issubdtype(input_seq.dtype, np.integer):
        raise AssertionError("Input must be an array of int, with values between 0 and the mapping size")
    # A negative index would silently wrap around to the end of the mapping (e.g. an EOS of -1 read as a char)
    if len(input_seq) > 0 and (min(input_seq) < 0 or max(input_seq) >= len(mapping)):
        raise AssertionError("Input must be an array of int, with values between 0 and the mapping size")

    return ("" if isinstance(mapping, str) else " ").join(map(mapping.__getitem__, input_seq))


def encode_sequences(
    sequences: list[str],
    vocab: str | Sequence[str],
    target_size: int | None = None,
    eos: int = -1,
    sos: int | None = None,
    pad: int | None = None,
    dynamic_seq_length: bool = False,
) -> np.ndarray:
    """Encode character sequences using a given vocab as mapping

    Args:
        sequences: the list of character sequences of size N
        vocab: the ordered vocab to use for encoding: a string of characters, or a list of tokens (cf.
            :func:`encode_string`)
        target_size: maximum length of the encoded data
        eos: encoding of End Of String
        sos: optional encoding of Start Of String
        pad: optional encoding for padding. In case of padding, all sequences are followed by 1 EOS then PAD
        dynamic_seq_length: if `target_size` is specified, uses it as upper bound and enables dynamic sequence size

    Returns:
        the padded encoded data as a tensor
    """
    if 0 <= eos < len(vocab):
        raise ValueError("argument 'eos' needs to be outside of vocab possible indices")

    encoded_seqs = [encode_string(seq, vocab) for seq in sequences]
    if not isinstance(target_size, int) or dynamic_seq_length:
        # Maximum sequence length (in vocab units) + EOS
        max_length = max((len(seq) for seq in encoded_seqs), default=0) + 1
        if isinstance(sos, int):
            max_length += 1
        if isinstance(pad, int):
            max_length += 1
        target_size = max_length if not isinstance(target_size, int) else min(max_length, target_size)

    # Pad all sequences
    if isinstance(pad, int):  # pad with padding symbol
        if 0 <= pad < len(vocab):
            raise ValueError("argument 'pad' needs to be outside of vocab possible indices")
        # In that case, add EOS at the end of the word before padding
        default_symbol = pad
    else:  # pad with eos symbol
        default_symbol = eos
    encoded_data: np.ndarray = np.full([len(sequences), target_size], default_symbol, dtype=np.int32)

    # Encode the strings
    for idx, seq in enumerate(encoded_seqs):
        if isinstance(pad, int):  # add eos at the end of the sequence
            seq.append(eos)
        encoded_data[idx, : min(len(seq), target_size)] = seq[: min(len(seq), target_size)]

    if isinstance(sos, int):  # place sos symbol at the beginning of each sequence
        if 0 <= sos < len(vocab):
            raise ValueError("argument 'sos' needs to be outside of vocab possible indices")
        encoded_data = np.roll(encoded_data, 1)
        encoded_data[:, 0] = sos

    return encoded_data


def convert_target_to_relative(
    img: ImageTensor, target: np.ndarray | dict[str, Any]
) -> tuple[ImageTensor, dict[str, Any] | np.ndarray]:
    """Converts target to relative coordinates

    Args:
        img: tf.Tensor or torch.Tensor representing the image
        target: target to convert to relative coordinates (boxes (N, 4) or polygons (N, 4, 2))

    Returns:
        The image and the target in relative coordinates
    """
    if isinstance(target, np.ndarray):
        target = convert_to_relative_coords(target, get_img_shape(img))  # type: ignore[arg-type]
    else:
        target["boxes"] = convert_to_relative_coords(target["boxes"], get_img_shape(img))  # type: ignore[arg-type]
    return img, target


def crop_bboxes_from_image(img_path: str | Path, geoms: np.ndarray) -> list[np.ndarray]:
    """Crop a set of bounding boxes from an image

    Args:
        img_path: path to the image
        geoms: a array of polygons of shape (N, 4, 2) or of straight boxes of shape (N, 4)

    Returns:
        a list of cropped images
    """
    with Image.open(img_path) as pil_img:
        img: np.ndarray = np.asarray(pil_img.convert("RGB"))
    # Polygon
    if geoms.ndim == 3 and geoms.shape[1:] == (4, 2):
        return extract_rcrops(img, geoms.astype(dtype=int))
    if geoms.ndim == 2 and geoms.shape[1] == 4:
        return extract_crops(img, geoms.astype(dtype=int))
    raise ValueError("Invalid geometry format")


def pre_transform_multiclass(
    img, target: tuple[np.ndarray, list], class_names: list[str] | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Converts multiclass target to relative coordinates.

    Args:
        img: Image
        target: tuple of target polygons and their classes names
        class_names: if given, the returned dictionary contains exactly these classes, in this order.
            Classes without any box in this sample map to an empty array. This keeps the class -> channel
            mapping stable across samples, which the detection models rely on.

    Returns:
        Image and dictionary of boxes, with class names as keys
    """
    boxes = convert_to_relative_coords(target[0], get_img_shape(img))
    boxes_classes = target[1]
    keys = list(class_names) if class_names is not None else sorted(set(boxes_classes))
    boxes_dict: dict[str, list] = {k: [] for k in keys}
    for k, poly in zip(boxes_classes, boxes):
        if k not in boxes_dict:
            raise ValueError(f"unknown class '{k}', expected one of {keys}")
        boxes_dict[k].append(poly)
    return img, {
        k: np.stack(v, axis=0) if v else np.zeros((0, *boxes.shape[1:]), dtype=boxes.dtype)
        for k, v in boxes_dict.items()
    }

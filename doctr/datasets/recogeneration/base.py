# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, Callable, List, Optional, Tuple
from random import SystemRandom

import cv2
import numpy as np
from PIL import Image, ImageDraw

from doctr.io.image import tensor_from_pil
from doctr.utils.fonts import get_font

from ..datasets import AbstractDataset
from ..utils import encode_sequences
from ..vocabs import VOCABS

save_rnd = SystemRandom()
os.environ["PYTHONIOENCODING"] = "utf-8"  # TODO: remove this for example with a set of language supported fonts ?


def generate_random_string(lang: str, length: int, variable_length: bool) -> str:
    """Generate a random string of the given length

    Args:
        lang: the language vocab of the string
        length: the length of the string
        variable_length: whether the length of the string is variable

    Returns:
        a random string
    """

    if lang not in VOCABS.keys():
        raise ValueError(f'unknown language: {lang}\nchoose one from: {VOCABS.keys()}')

    vocab = VOCABS[lang]
    rnd_string = ''
    rnd_string += "".join([save_rnd.choice(vocab)
                          for _ in range(0, save_rnd.randint(1, length) if variable_length else length)])
    return rnd_string


def synthesize_word_img(word: str, size: tuple, background: int,
                        font_size: int, font_family: Optional[str] = None) -> Image:
    """Generate a synthetic word image with white / noise background and black text

    Args:
        word: the word to render as an image
        size: the size of the rendered image (height, width)
        background: the background color of the image (0 - white, 1 - noise, 2 - random)
        font_size: the font size of the rendered image
        font_family: the font family (has to be installed on your system)

    Returns:
        PIL image of the character
    """

    if background == 2:
        background = save_rnd.randint(0, 1)

    if background == 0:
        img = Image.new("L", (size[1], size[0]), 255).convert("RGB")
    elif background == 1:
        image = np.ones((size[0], size[1])) * 255
        cv2.randn(image, 235, 10)
        img = Image.fromarray(image).convert("RGB")
    else:
        raise ValueError(f'unknown background: {background}\nchoose one from: 0 - white, 1 - noisy, 2 - random')

    d = ImageDraw.Draw(img)

    # Draw the word centered in the image
    font = get_font(font_family, font_size)
    w, h = d.textsize(word, font)

    if w > size[1] or h > size[0]:
        raise ValueError(
            f'word "{word}" is too large for the given size ({size[1]}, {size[0]}): \
                decrease fontsize or increase image size')

    d.text(((size[1] - w) / 2, (size[0] - h) / 2), word, font=font, fill=(0, 0, 0))

    return img


class _WordGenerator(AbstractDataset):

    def __init__(
        self,
        lang: str,
        num_samples: int,
        max_length: int = 10,
        size: tuple = (48, 160),
        background: int = 0,
        variable_length: bool = False,
        font_family: Optional[str] = None,
        font_size: int = 20,
        cache_samples: bool = False,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.sample_transforms = sample_transforms
        self._num_samples = num_samples
        self._size = size
        self._background = background
        self._font_family = font_family
        self._font_size = font_size
        self.vocab = VOCABS

        self._data: List[Image.Image] = []
        self._words = [
            generate_random_string(lang,
                                   length=max_length,
                                   variable_length=variable_length)
            for _ in range(self._num_samples)
        ]
        self._targets = encode_sequences(self._words, self.vocab[lang], pad=-100)
        if cache_samples:
            self._data = [
                synthesize_word_img(word,
                                    size=self._size,
                                    background=self._background,
                                    font_size=font_size,
                                    font_family=self._font_family)
                for word in self._words
            ]

    def __len__(self) -> int:
        return self._num_samples

    def _read_sample(self, index: int) -> Tuple[Any, np.ndarray]:
        target = index % len(self._words)
        # Samples are already cached
        if len(self._data) > 0:
            pil_img = self._data[target].copy()
        else:
            pil_img = synthesize_word_img(self._words[target],
                                          size=self._size,
                                          background=self._background,
                                          font_size=self._font_size,
                                          font_family=self._font_family)
        img = tensor_from_pil(pil_img)

        return img, self._targets[target]

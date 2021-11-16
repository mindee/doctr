# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any, Callable, List, Optional, Tuple
import random

from PIL import Image, ImageDraw

from doctr.io.image import tensor_from_pil
from doctr.utils.fonts import get_font

from ..datasets import AbstractDataset
from ..vocabs import VOCABS


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
    rnd_string += "".join([random.choice(vocab)
                          for _ in range(0, random.randint(1, length) if variable_length else length)])
    return rnd_string


def synthesize_word_img(word: str, size: tuple, font_size: int = 18, font_family: Optional[str] = None) -> Image:
    """Generate a synthetic word image with white background and black text

    Args:
        word: the word to render as an image
        size: the size of the rendered image
        font_family: the font family (has to be installed on your system)

    Returns:
        PIL image of the character
    """

    img = Image.new('RGB', (size[1], size[0]), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Draw the word centered in the image
    font = get_font(font_family, font_size)
    w, h = d.textsize(word, font)

    if w > size[1] or h > size[0]:
        raise ValueError(f'word "{word}" is too large for the given size ({size[1]}, {size[0]})')

    d.text(((size[1] - w) / 2, (size[0] - h) / 2), word, font=font, fill=(0, 0, 0))

    #img.save(f'1.png')

    return img


class _WordGenerator(AbstractDataset):

    def __init__(
        self,
        lang: str,
        num_samples: int,
        max_length: int = 10,
        size: tuple = (48, 160),
        font_size: int = 18,
        variable_length: bool = False,
        cache_samples: bool = False,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
        font_family: Optional[str] = None,
    ) -> None:
        self.sample_transforms = sample_transforms
        self.lang = lang
        self._num_samples = num_samples
        self.max_length = max_length
        self.size = size
        self.font_size = font_size
        self.variable_length = variable_length
        self.font_family = font_family

        self._data: List[Image.Image] = []
        self.words = [generate_random_string(lang, length=self.max_length, variable_length=self.variable_length)
                      for _ in range(self._num_samples)]
        if cache_samples:
            self._data = [synthesize_word_img(word, size=self.size, font_size=self.font_size,
                                              font_family=self.font_family) for word in self.words]

    def __len__(self) -> int:
        return self._num_samples

    # TODO: handle targets
    def _read_sample(self, index: int) -> Tuple[Any, int]:
        target = index % len(self.words)
        # Samples are already cached
        if len(self._data) > 0:
            pil_img = self._data[target].copy()
        else:
            pil_img = synthesize_word_img(self.words[target], size=self.size, font_family=self.font_family)
        img = tensor_from_pil(pil_img)

        return img, self.words[target]

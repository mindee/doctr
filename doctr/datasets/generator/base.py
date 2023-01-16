# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image, ImageDraw

from doctr.io.image import tensor_from_pil
from doctr.utils.fonts import get_font

from ..datasets import AbstractDataset


def synthesize_text_img(
    text: str,
    font_size: int = 32,
    font_family: Optional[str] = None,
    background_color: Optional[Tuple[int, int, int]] = None,
    text_color: Optional[Tuple[int, int, int]] = None,
) -> Image:
    """Generate a synthetic text image

    Args:
        text: the text to render as an image
        font_size: the size of the font
        font_family: the font family (has to be installed on your system)
        background_color: background color of the final image
        text_color: text color on the final image

    Returns:
        PIL image of the text
    """

    background_color = (0, 0, 0) if background_color is None else background_color
    text_color = (255, 255, 255) if text_color is None else text_color

    font = get_font(font_family, font_size)
    text_w, text_h = font.getsize(text)
    h, w = int(round(1.3 * text_h)), int(round(1.1 * text_w))
    # If single letter, make the image square, otherwise expand to meet the text size
    img_size = (h, w) if len(text) > 1 else (max(h, w), max(h, w))

    img = Image.new("RGB", img_size[::-1], color=background_color)
    d = ImageDraw.Draw(img)

    # Offset so that the text is centered
    text_pos = (int(round((img_size[1] - text_w) / 2)), int(round((img_size[0] - text_h) / 2)))
    # Draw the text
    d.text(text_pos, text, font=font, fill=text_color)
    return img


class _CharacterGenerator(AbstractDataset):
    def __init__(
        self,
        vocab: str,
        num_samples: int,
        cache_samples: bool = False,
        font_family: Optional[Union[str, List[str]]] = None,
        img_transforms: Optional[Callable[[Any], Any]] = None,
        sample_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ) -> None:
        self.vocab = vocab
        self._num_samples = num_samples
        self.font_family = font_family if isinstance(font_family, list) else [font_family]  # type: ignore[list-item]
        # Validate fonts
        if isinstance(font_family, list):
            for font in self.font_family:
                try:
                    _ = get_font(font, 10)
                except OSError:
                    raise ValueError(f"unable to locate font: {font}")
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms

        self._data: List[Image.Image] = []
        if cache_samples:
            self._data = [
                (synthesize_text_img(char, font_family=font), idx)
                for idx, char in enumerate(self.vocab)
                for font in self.font_family
            ]

    def __len__(self) -> int:
        return self._num_samples

    def _read_sample(self, index: int) -> Tuple[Any, int]:
        # Samples are already cached
        if len(self._data) > 0:
            idx = index % len(self._data)
            pil_img, target = self._data[idx]
        else:
            target = index % len(self.vocab)
            pil_img = synthesize_text_img(self.vocab[target], font_family=random.choice(self.font_family))
        img = tensor_from_pil(pil_img)

        return img, target


class _WordGenerator(AbstractDataset):
    def __init__(
        self,
        vocab: str,
        min_chars: int,
        max_chars: int,
        num_samples: int,
        cache_samples: bool = False,
        font_family: Optional[Union[str, List[str]]] = None,
        img_transforms: Optional[Callable[[Any], Any]] = None,
        sample_transforms: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ) -> None:
        self.vocab = vocab
        self.wordlen_range = (min_chars, max_chars)
        self._num_samples = num_samples
        self.font_family = font_family if isinstance(font_family, list) else [font_family]  # type: ignore[list-item]
        # Validate fonts
        if isinstance(font_family, list):
            for font in self.font_family:
                try:
                    _ = get_font(font, 10)
                except OSError:
                    raise ValueError(f"unable to locate font: {font}")
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms

        self._data: List[Image.Image] = []
        if cache_samples:
            _words = [self._generate_string(*self.wordlen_range) for _ in range(num_samples)]
            self._data = [
                (synthesize_text_img(text, font_family=random.choice(self.font_family)), text) for text in _words
            ]

    def _generate_string(self, min_chars: int, max_chars: int) -> str:
        num_chars = random.randint(min_chars, max_chars)
        return "".join(random.choice(self.vocab) for _ in range(num_chars))

    def __len__(self) -> int:
        return self._num_samples

    def _read_sample(self, index: int) -> Tuple[Any, str]:
        # Samples are already cached
        if len(self._data) > 0:
            pil_img, target = self._data[index]
        else:
            target = self._generate_string(*self.wordlen_range)
            pil_img = synthesize_text_img(target, font_family=random.choice(self.font_family))
        img = tensor_from_pil(pil_img)

        return img, target

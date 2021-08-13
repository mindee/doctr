# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from PIL import ImageDraw, Image, ImageFont
from typing import Any, Tuple, Optional, Callable

from doctr.io.image import tensor_from_pil
from ..datasets import AbstractDataset


def synthesize_char_img(char: str, size: int = 32, font_family: str = "FreeMono.ttf") -> Image:
    img = Image.new('RGB', (size, size), color=(0, 0, 0))
    d = ImageDraw.Draw(img)

    # Draw the character
    font = ImageFont.truetype(font_family, size)
    d.text((4, 0), char, font=font, fill=(255, 255, 255))

    return img


class _CharacterGenerator(AbstractDataset):
    """Implements a character image generation dataset

    Args:
        vocab: vocabulary to take the character from
        num_samples: number of samples that will be generated iterating over the dataset
        cache_samples: whether generated images should be cached firsthand
        sample_transforms: composable transformations that will be applied to each image
    """

    def __init__(
        self,
        vocab: str,
        num_samples: int,
        cache_samples: bool = False,
        sample_transforms: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.sample_transforms = sample_transforms
        self.vocab = vocab
        self._num_samples = num_samples

        self._data = [synthesize_char_img(char) for char in self.vocab] if cache_samples else []

    def __len__(self) -> int:
        return self._num_samples

    def _read_sample(self, index: int) -> Tuple[Any, int]:
        target = index % len(self.vocab)
        # Samples are already cached
        if len(self._data) > 0:
            pil_img = self._data[target].copy()
        else:
            pil_img = synthesize_char_img(self.vocab[target])
        img = tensor_from_pil(pil_img)

        return img, target

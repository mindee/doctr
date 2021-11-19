# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from torch.utils.data._utils.collate import default_collate

from .base import _WordGenerator

__all__ = ['WordGenerator']


class WordGenerator(_WordGenerator):
    """Implements a word image generation dataset

    Example::
        >>> from doctr.datasets import WordGenerator
        >>> ds = WordGenerator(lang='english', num_samples=10)
        >>> img, target = ds[0]

    Args:
        lang: language vocabulary to take the characters from
        num_samples: number of samples that will be generated iterating over the dataset
        max_length: maximum length of the generated word
        size: size of the generated image (height, width)
        background: background color of the generated image (0 - white, 1 - noise, 2 - random)
        variable_length: whether to allow variable length words in range of [1, max_length]
        font_family: font family to use for the generated image
        font_size: font size to use for the generated image
        cache_samples: whether generated images should be cached firsthand
        sample_transforms: composable transformations that will be applied to each image
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        setattr(self, 'collate_fn', default_collate)

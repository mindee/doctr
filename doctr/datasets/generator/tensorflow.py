# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import tensorflow as tf

from .base import _CharacterGenerator, _WordGenerator

__all__ = ["CharacterGenerator", "WordGenerator"]


class CharacterGenerator(_CharacterGenerator):
    """Implements a character image generation dataset

    >>> from doctr.datasets import CharacterGenerator
    >>> ds = CharacterGenerator(vocab='abdef', num_samples=100)
    >>> img, target = ds[0]

    Args:
        vocab: vocabulary to take the character from
        num_samples: number of samples that will be generated iterating over the dataset
        cache_samples: whether generated images should be cached firsthand
        font_family: font to use to generate the text images
        img_transforms: composable transformations that will be applied to each image
        sample_transforms: composable transformations that will be applied to both the image and the target
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(samples):
        images, targets = zip(*samples)
        images = tf.stack(images, axis=0)

        return images, tf.convert_to_tensor(targets)


class WordGenerator(_WordGenerator):
    """Implements a character image generation dataset

    >>> from doctr.datasets import WordGenerator
    >>> ds = WordGenerator(vocab='abdef', min_chars=1, max_chars=32, num_samples=100)
    >>> img, target = ds[0]

    Args:
        vocab: vocabulary to take the character from
        min_chars: minimum number of characters in a word
        max_chars: maximum number of characters in a word
        num_samples: number of samples that will be generated iterating over the dataset
        cache_samples: whether generated images should be cached firsthand
        font_family: font to use to generate the text images
        img_transforms: composable transformations that will be applied to each image
        sample_transforms: composable transformations that will be applied to both the image and the target
    """

    pass

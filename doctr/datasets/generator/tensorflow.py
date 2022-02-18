# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import tensorflow as tf

from doctr import transforms as T
from doctr.transforms.functional import rotated_img_tensor

from .base import _CharacterGenerator, _WordGenerator

__all__ = ['CharacterGenerator', 'WordGenerator']


class CharacterGenerator(_CharacterGenerator):
    """Implements a character image generation dataset

    Example::
        >>> from doctr.datasets import CharacterGenerator
        >>> ds = CharacterGenerator(vocab='abdef')
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

    Example::
        >>> from doctr.datasets import WordGenerator
        >>> ds = WordGenerator(vocab='abdef')
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.img_transforms is None:
            img_transforms = T.Compose([
                T.RandomApply(T.LambdaTransformation(
                    lambda x: rotated_img_tensor(x, np.random.choice([-6.0, 6.0]), expand=True)), .2),
                T.Resize((32, 128), preserve_aspect_ratio=True),
                T.RandomApply(T.ColorInversion(min_val=1.0), 1.0),
                T.RandomApply(T.GaussianBlur(
                    kernel_shape=3, std=(0.3, 2.0)), .2),
            ])
            setattr(self, 'img_transforms', img_transforms)

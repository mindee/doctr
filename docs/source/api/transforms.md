# doctr.transforms

```{eval-rst}
.. module:: doctr.transforms
```

Data transformations for training and inference. Inspired by
[torchvision](https://github.com/pytorch/vision), they are expressed as composable modules. Image transforms apply to
the image only; sample transforms (wrapped in {class}`SampleCompose`) also update the target, e.g. the boxes of a
rotated image.

```python
from torchvision.transforms.v2 import Compose

from doctr import transforms as T

img_transforms = Compose([
    T.Resize((32, 128), preserve_aspect_ratio=True),
    T.RandomApply(T.ColorInversion(), 0.1),
    T.RandomApply(T.GaussianNoise(mean=0.1, std=0.1), 0.1),
])
```

## Image transformations

```{eval-rst}
.. autoclass:: Resize

.. autoclass:: RandomResize

.. autoclass:: GaussianNoise

.. autoclass:: GaussianBlur

.. autoclass:: ChannelShuffle

.. autoclass:: ColorInversion

.. autoclass:: RandomShadow

.. autoclass:: RandomGlare

.. autoclass:: RandomLighting
```

## Geometric transformations

```{eval-rst}
.. autoclass:: RandomHorizontalFlip

.. autoclass:: RandomRotate

.. autoclass:: RandomCrop

.. autoclass:: RandomPerspective
```

## Composition

```{eval-rst}
.. autoclass:: SampleCompose

.. autoclass:: ImageTransform

.. autoclass:: ImageTorchvisionTransform

.. autoclass:: OneOf

.. autoclass:: RandomApply
```

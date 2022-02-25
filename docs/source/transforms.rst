doctr.transforms
================

.. currentmodule:: doctr.transforms

Data transformations are part of both training and inference procedure. Drawing inspiration from the design of `torchvision <https://github.com/pytorch/vision>`_, we express transformations as composable modules.


Supported transformations
-------------------------
Here are all transformations that are available through docTR:

.. autoclass:: Resize
.. autoclass:: Normalize
.. autoclass:: LambdaTransformation
.. autoclass:: ToGray
.. autoclass:: ColorInversion
.. autoclass:: RandomBrightness
.. autoclass:: RandomContrast
.. autoclass:: RandomSaturation
.. autoclass:: RandomHue
.. autoclass:: RandomGamma
.. autoclass:: RandomJpegQuality
.. autoclass:: RandomRotate
.. autoclass:: RandomCrop
.. autoclass:: GaussianBlur
.. autoclass:: ChannelShuffle
.. autoclass:: GaussianNoise
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomShadow


Composing transformations
---------------------------------------------
It is common to require several transformations to be performed consecutively.

.. autoclass:: Compose
.. autoclass:: OneOf
.. autoclass:: RandomApply

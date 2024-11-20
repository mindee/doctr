doctr.transforms
================

.. currentmodule:: doctr.transforms

Data transformations are part of both training and inference procedure. Drawing inspiration from the design of `torchvision <https://github.com/pytorch/vision>`_, we express transformations as composable modules.


Supported transformations
-------------------------
Here are all transformations that are available through docTR:

.. currentmodule:: doctr.transforms.modules

.. autoclass:: Resize
.. autoclass:: GaussianNoise
.. autoclass:: ChannelShuffle
.. autoclass:: RandomHorizontalFlip
.. autoclass:: RandomShadow
.. autoclass:: RandomResize


Composing transformations
---------------------------------------------
It is common to require several transformations to be performed consecutively.

.. autoclass:: SampleCompose
.. autoclass:: ImageTransform
.. autoclass:: ColorInversion
.. autoclass:: OneOf
.. autoclass:: RandomApply
.. autoclass:: RandomRotate
.. autoclass:: RandomCrop

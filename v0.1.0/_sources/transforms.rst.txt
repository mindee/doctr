doctr.transforms
================

.. currentmodule:: doctr.transforms

Data transformations are part of both training and inference procedure. Drawing inspiration from the design of `torchvision <https://github.com/pytorch/vision>`_, we express transformations as composable modules.


Supported transformations
-------------------------
Here are all transformations that are available through DocTR:

.. autoclass:: Resize
.. autoclass:: Normalize
.. autoclass:: LambdaTransformation


Composing several consecutive transformations
---------------------------------------------
It is common to require several transformations to be performed consecutively.

.. autoclass:: Compose

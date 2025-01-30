AWS Lambda
==========

The security policy of `AWS Lambda <https://aws.amazon.com/lambda/>`_ restricts writing outside the ``/tmp`` directory.

To make docTR work on Lambda, you need to perform the following two steps:

1. Disable the usage of the ``multiprocessing`` package by setting the ``DOCTR_MULTIPROCESSING_DISABLE`` environment variable to ``TRUE``. This step is necessary because the package uses the ``/dev/shm`` directory for shared memory.

2. Change the caching directory used by docTR for models. By default, it is set to ``~/.cache/doctr``, which is outside the ``/tmp`` directory on AWS Lambda. You can modify this by setting the ``DOCTR_CACHE_DIR`` environment variable.

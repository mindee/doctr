doctr.datasets
==============

.. currentmodule:: doctr.datasets

Whether it is for training or for evaluation, having predefined objects to access datasets in your prefered framework
can be a significant save of time.


.. _datasets:

Available Datasets
------------------
Here are all datasets that are available through docTR:

.. autoclass:: FUNSD
.. autoclass:: SROIE
.. autoclass:: CORD
.. autoclass:: OCRDataset
.. autoclass:: CharacterGenerator
.. autoclass:: DocArtefacts
.. autoclass:: IIIT5K
.. autoclass:: SVT
.. autoclass:: SVHN
.. autoclass:: SynthText
.. autoclass:: IC03
.. autoclass:: IC13


Data Loading
------------
Each dataset has its specific way to load a sample, but handling batch aggregation and the underlying iterator is a task deferred to another object in docTR.

.. autoclass:: doctr.datasets.loader.DataLoader


.. _vocabs:

Supported Vocabs
----------------

Since textual content has to be encoded properly for models to interpret them efficiently, docTR supports multiple sets
of vocabs.

.. list-table:: docTR Vocabs
   :widths: 20 5 50
   :header-rows: 1

   * - Name
     - size
     - characters
   * - digits
     - 10
     - 0123456789
   * - ascii_letters
     - 52
     - abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
   * - punctuation
     - 32
     - !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
   * - currency
     - 5
     - £€¥¢฿
   * - latin
     - 94
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
   * - english
     - 100
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - legacy_french
     - 123
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~°àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ£€¥¢฿
   * - french
     - 126
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ
   * - portuguese
     - 131
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áàâãéêëíïóôõúüçÁÀÂÃÉËÍÏÓÔÕÚÜÇ¡¿
   * - spanish
     - 116
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áéíóúüñÁÉÍÓÚÜÑ¡¿
   * - german
     - 108
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿äöüßÄÖÜẞ

.. autofunction:: encode_sequences

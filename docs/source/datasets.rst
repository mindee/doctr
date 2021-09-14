doctr.datasets
==============

.. currentmodule:: doctr.datasets

Whether it is for training or for evaluation, having predefined objects to access datasets in your prefered framework
can be a significant save of time.


.. _datasets:

Available Datasets
------------------
The datasets from DocTR inherit from an abstract class that handles verified downloading from a given URL.

.. autoclass:: doctr.datasets.datasets.VisionDataset


Here are all datasets that are available through DocTR:

.. autoclass:: FUNSD
.. autoclass:: SROIE
.. autoclass:: CORD
.. autoclass:: OCRDataset
.. autoclass:: CharacterGenerator


Data Loading
------------
Each dataset has its specific way to load a sample, but handling batch aggregation and the underlying iterator is a task deferred to another object in DocTR.

.. autoclass:: doctr.datasets.loader.DataLoader


.. _vocabs:

Supported Vocabs
----------------

Since textual content has to be encoded properly for models to interpret them efficiently, DocTR supports multiple sets
of vocabs.

.. list-table:: DocTR Vocabs
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

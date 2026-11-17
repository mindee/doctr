# doctr.datasets

```{eval-rst}
.. module:: doctr.datasets
```

Public datasets, synthetic data generators and loaders for your own annotations. See [](../training/datasets.md)
for a guide.

## Public datasets

```{eval-rst}
.. autoclass:: FUNSD

.. autoclass:: SROIE

.. autoclass:: CORD

.. autoclass:: IIIT5K

.. autoclass:: SVT

.. autoclass:: SVHN

.. autoclass:: SynthText

.. autoclass:: IC03

.. autoclass:: IC13

.. autoclass:: IMGUR5K

.. autoclass:: MJSynth

.. autoclass:: IIITHWS

.. autoclass:: WILDRECEIPT

.. autoclass:: COCOTEXT

.. autoclass:: DocArtefacts
```

## Synthetic data generators

```{eval-rst}
.. autoclass:: CharacterGenerator

.. autoclass:: WordGenerator
```

## Custom datasets

```{eval-rst}
.. autoclass:: DetectionDataset

.. autoclass:: RecognitionDataset

.. autoclass:: OCRDataset

.. autoclass:: LayoutDataset

.. autoclass:: TableStructureDataset

.. autoclass:: OrientationDataset
```

## Vocabularies

```{eval-rst}
.. data:: VOCABS
   :type: dict[str, str]

   The supported vocabularies, by name. See :doc:`../models/vocabs` for the complete list.

.. data:: LATEX_TOKENS
   :type: list[str]

   The 1,324 LaTeX tokens of formula recognition: the printable ASCII characters, the control symbols, the commands and
   environments supported by KaTeX, and the other commands and characters found in public formula datasets (UniMER-1M,
   HME100K, UniMER-Test). See :doc:`../models/formula`.
```

## Utilities

```{eval-rst}
.. autofunction:: translate

.. autofunction:: tokenize_latex

.. autofunction:: normalize_latex

.. autofunction:: encode_string

.. autofunction:: decode_sequence

.. autofunction:: encode_sequences

.. autofunction:: pre_transform_multiclass

.. autofunction:: crop_bboxes_from_image

.. autofunction:: convert_target_to_relative
```

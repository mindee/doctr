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
```

## Utilities

```{eval-rst}
.. autofunction:: translate

.. autofunction:: encode_string

.. autofunction:: decode_sequence

.. autofunction:: encode_sequences

.. autofunction:: pre_transform_multiclass

.. autofunction:: crop_bboxes_from_image

.. autofunction:: convert_target_to_relative
```

# doctr.models

```{eval-rst}
.. module:: doctr.models
```

Predictors and model architectures. All the factory functions below are also importable from `doctr.models`
directly. See the [model zoo](../models/index.md) for benchmarks.

## End-to-end predictors

```{eval-rst}
.. autofunction:: doctr.models.ocr_predictor

.. autofunction:: doctr.models.kie_predictor

.. autoclass:: doctr.models.predictor.OCRPredictor
   :members: add_hook, stream

.. autoclass:: doctr.models.kie_predictor.KIEPredictor
   :members: add_hook, stream

.. autoclass:: doctr.models.builder.DocumentBuilder

.. autoclass:: doctr.models.preprocessor.PreProcessor
```

## Text detection

```{eval-rst}
.. autofunction:: doctr.models.detection.detection_predictor

.. autofunction:: doctr.models.detection.db_resnet34

.. autofunction:: doctr.models.detection.db_resnet50

.. autofunction:: doctr.models.detection.db_mobilenet_v3_large

.. autofunction:: doctr.models.detection.linknet_resnet18

.. autofunction:: doctr.models.detection.linknet_resnet34

.. autofunction:: doctr.models.detection.linknet_resnet50

.. autofunction:: doctr.models.detection.fast_tiny

.. autofunction:: doctr.models.detection.fast_small

.. autofunction:: doctr.models.detection.fast_base
```

## Text recognition

```{eval-rst}
.. autofunction:: doctr.models.recognition.recognition_predictor

.. autofunction:: doctr.models.recognition.crnn_vgg16_bn

.. autofunction:: doctr.models.recognition.crnn_mobilenet_v3_small

.. autofunction:: doctr.models.recognition.crnn_mobilenet_v3_large

.. autofunction:: doctr.models.recognition.sar_resnet31

.. autofunction:: doctr.models.recognition.master

.. autofunction:: doctr.models.recognition.vitstr_small

.. autofunction:: doctr.models.recognition.vitstr_base

.. autofunction:: doctr.models.recognition.parseq

.. autofunction:: doctr.models.recognition.viptr_tiny
```

## Layout analysis

```{eval-rst}
.. autofunction:: doctr.models.layout.layout_predictor

.. autofunction:: doctr.models.layout.lw_detr_s

.. autofunction:: doctr.models.layout.lw_detr_m
```

## Table structure recognition

```{eval-rst}
.. autofunction:: doctr.models.table_structure.table_predictor

.. autofunction:: doctr.models.table_structure.tablecenternet
```

## Classification

Orientation classifiers, and the backbones used by the detection and recognition models.

```{eval-rst}
.. autofunction:: doctr.models.classification.crop_orientation_predictor

.. autofunction:: doctr.models.classification.page_orientation_predictor

.. autofunction:: doctr.models.classification.mobilenet_v3_small_crop_orientation

.. autofunction:: doctr.models.classification.mobilenet_v3_small_page_orientation

.. autofunction:: doctr.models.classification.vgg16_bn_r

.. autofunction:: doctr.models.classification.resnet18

.. autofunction:: doctr.models.classification.resnet31

.. autofunction:: doctr.models.classification.resnet34

.. autofunction:: doctr.models.classification.resnet50

.. autofunction:: doctr.models.classification.magc_resnet31

.. autofunction:: doctr.models.classification.mobilenet_v3_small

.. autofunction:: doctr.models.classification.mobilenet_v3_large

.. autofunction:: doctr.models.classification.mobilenet_v3_small_r

.. autofunction:: doctr.models.classification.mobilenet_v3_large_r

.. autofunction:: doctr.models.classification.vit_s

.. autofunction:: doctr.models.classification.vit_b

.. autofunction:: doctr.models.classification.textnet_tiny

.. autofunction:: doctr.models.classification.textnet_small

.. autofunction:: doctr.models.classification.textnet_base

.. autofunction:: doctr.models.classification.vip_tiny

.. autofunction:: doctr.models.classification.vip_base

.. autofunction:: doctr.models.classification.vit_det_s

.. autofunction:: doctr.models.classification.vit_det_m

.. autofunction:: doctr.models.classification.starnet_s3
```

## Hugging Face Hub

```{eval-rst}
.. autofunction:: doctr.models.factory.from_hub

.. autofunction:: doctr.models.factory.push_to_hf_hub

.. autofunction:: doctr.models.factory.login_to_hub
```

## Model utilities

```{eval-rst}
.. autofunction:: doctr.models.utils.add_whitelist

.. autofunction:: doctr.models.utils.export_model_to_onnx
```

## Reading order

Language-independent estimation of the reading order of the elements of a page. It combines a column-following
topological sort of their spatial relations with optional layout semantics (headers, footers, captions), and is
used by all the [exports](../user_guide/exporting.md#reading-order).

```{eval-rst}
.. module:: doctr.models.reading_order

.. autoclass:: doctr.models.reading_order.ReadingOrderPredictor

.. autofunction:: doctr.models.reading_order.sort_reading_order

.. autofunction:: doctr.models.reading_order.detect_text_direction

.. autofunction:: doctr.models.reading_order.resolve_reading_segments

.. autofunction:: doctr.models.reading_order.assign_layout_labels

.. autofunction:: doctr.models.reading_order.deskew_reading_geometries
```

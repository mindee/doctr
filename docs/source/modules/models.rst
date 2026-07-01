doctr.models
============

.. currentmodule:: doctr.models


doctr.models.classification
---------------------------

.. autofunction:: doctr.models.classification.vgg16_bn_r

.. autofunction:: doctr.models.classification.resnet18

.. autofunction:: doctr.models.classification.resnet34

.. autofunction:: doctr.models.classification.resnet50

.. autofunction:: doctr.models.classification.resnet31

.. autofunction:: doctr.models.classification.mobilenet_v3_small

.. autofunction:: doctr.models.classification.mobilenet_v3_large

.. autofunction:: doctr.models.classification.mobilenet_v3_small_r

.. autofunction:: doctr.models.classification.mobilenet_v3_large_r

.. autofunction:: doctr.models.classification.mobilenet_v3_small_crop_orientation

.. autofunction:: doctr.models.classification.mobilenet_v3_small_page_orientation

.. autofunction:: doctr.models.classification.magc_resnet31

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

.. autofunction:: doctr.models.classification.crop_orientation_predictor

.. autofunction:: doctr.models.classification.page_orientation_predictor


doctr.models.detection
----------------------

.. autofunction:: doctr.models.detection.linknet_resnet18

.. autofunction:: doctr.models.detection.linknet_resnet34

.. autofunction:: doctr.models.detection.linknet_resnet50

.. autofunction:: doctr.models.detection.db_resnet50

.. autofunction:: doctr.models.detection.db_mobilenet_v3_large

.. autofunction:: doctr.models.detection.fast_tiny

.. autofunction:: doctr.models.detection.fast_small

.. autofunction:: doctr.models.detection.fast_base

.. autofunction:: doctr.models.detection.detection_predictor


doctr.models.layout
-------------------

.. autofunction:: doctr.models.layout.lw_detr_s

.. autofunction:: doctr.models.layout.lw_detr_m

.. autofunction:: doctr.models.layout.layout_predictor


doctr.models.table_structure
----------------------------

.. autofunction:: doctr.models.table_structure.tablecenternet

.. autofunction:: doctr.models.table_structure.table_predictor


doctr.models.recognition
------------------------

.. autofunction:: doctr.models.recognition.crnn_vgg16_bn

.. autofunction:: doctr.models.recognition.crnn_mobilenet_v3_small

.. autofunction:: doctr.models.recognition.crnn_mobilenet_v3_large

.. autofunction:: doctr.models.recognition.sar_resnet31

.. autofunction:: doctr.models.recognition.master

.. autofunction:: doctr.models.recognition.vitstr_small

.. autofunction:: doctr.models.recognition.vitstr_base

.. autofunction:: doctr.models.recognition.parseq

.. autofunction:: doctr.models.recognition.viptr_tiny

.. autofunction:: doctr.models.recognition.recognition_predictor


doctr.models.zoo
----------------

.. autofunction:: doctr.models.ocr_predictor

.. autofunction:: doctr.models.kie_predictor


doctr.models.factory
--------------------

.. autofunction:: doctr.models.factory.login_to_hub

.. autofunction:: doctr.models.factory.from_hub

.. autofunction:: doctr.models.factory.push_to_hf_hub


doctr.models.utils
------------------

.. currentmodule:: doctr.models.utils

.. autofunction:: export_model_to_onnx

.. autofunction:: add_whitelist

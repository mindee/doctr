Train your own model
====================

If the pretrained models don't meet your specific needs, you have the option to train your own model using the docTR library.
For details on the training process and the necessary data and data format, refer to the following links:

- `detection <https://github.com/mindee/doctr/tree/main/references/detection#readme>`_
- `recognition <https://github.com/mindee/doctr/tree/main/references/recognition#readme>`_

If you’re looking for a lightweight yet efficient tool to annotate small amounts of data, especially tailored for docTR,
check out the `docTR Labeling Tool <https://github.com/text2knowledge/docTR-Labeler>`_.
This tool makes it easy to create your own dataset for fine-tuning and optimizing your OCR models.

Loading your custom trained model
---------------------------------

This section shows how you can easily load a custom trained model in docTR.

.. code:: python3

    import torch
    from doctr.models import ocr_predictor, db_resnet50, crnn_vgg16_bn

    # Load custom detection model
    det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
    det_model.from_pretrained('<path_to_pt>', map_location="cpu")
    predictor = ocr_predictor(det_arch=det_model, reco_arch="vitstr_small", pretrained=True)

    # Load custom recognition model
    reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
    reco_model.from_pretrained('<path_to_pt>', map_location="cpu")
    predictor = ocr_predictor(det_arch="linknet_resnet18", reco_arch=reco_model, pretrained=True)

    # Load custom detection and recognition model
    det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
    det_model.from_pretrained('<path_to_pt>', map_location="cpu")
    reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
    reco_model.from_pretrained('<path_to_pt>', map_location="cpu")
    predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model, pretrained=False)


Load a custom recognition model trained on another vocabulary as the default one (French):

.. code:: python3

    import torch
    from doctr.models import ocr_predictor, crnn_vgg16_bn
    from doctr.datasets import VOCABS

    reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["german"])
    reco_model.from_pretrained('<path_to_pt>')

    predictor = ocr_predictor(det_arch='linknet_resnet18', reco_arch=reco_model, pretrained=True)

Load a custom trained KIE detection model:

.. code:: python3

    import torch
    from doctr.models import kie_predictor, db_resnet50

    det_model = db_resnet50(pretrained=False, pretrained_backbone=False, class_names=['total', 'date'])
    det_model.from_pretrained('<path_to_pt>')
    kie_predictor(det_arch=det_model, reco_arch='crnn_vgg16_bn', pretrained=True)

Load a model with customized Preprocessor:

.. code:: python3

    import torch
    from doctr.models.predictor import OCRPredictor
    from doctr.models.detection.predictor import DetectionPredictor
    from doctr.models.recognition.predictor import RecognitionPredictor
    from doctr.models.preprocessor import PreProcessor
    from doctr.models import db_resnet50, crnn_vgg16_bn

    det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
    det_model.from_pretrained('<path_to_pt>')
    reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
    reco_model.from_pretrained('<path_to_pt>')

    det_predictor = DetectionPredictor(
        PreProcessor(
            (1024, 1024),
            batch_size=1,
            mean=(0.798, 0.785, 0.772),
            std=(0.264, 0.2749, 0.287)
        ),
        det_model
    )

    reco_predictor = RecognitionPredictor(
        PreProcessor(
            (32, 128),
            preserve_aspect_ratio=True,
            batch_size=32,
            mean=(0.694, 0.695, 0.693),
            std=(0.299, 0.296, 0.301)
        ),
        reco_model
    )

    predictor = OCRPredictor(det_predictor, reco_predictor)

Custom orientation classification models
----------------------------------------

If you work with rotated documents and make use of the orientation classification feature by passing one of the following arguments:

* `assume_straight_pages=False`
* `detect_orientation=True`
* `straigten_pages=True`

You can train your own orientation classification model using the docTR library. For details on the training process and the necessary data and data format, refer to the following link:

- `orientation <https://github.com/mindee/doctr/blob/main/references/classification/README.md#usage-orientation-classification>`_

**NOTE**: Currently we support only `mobilenet_v3_small` models for crop and page orientation classification.

Loading your custom trained orientation classification model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python3

    import torch
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor, mobilenet_v3_small_page_orientation, mobilenet_v3_small_crop_orientation
    from doctr.models.classification.zoo import crop_orientation_predictor, page_orientation_predictor

    custom_page_orientation_model = mobilenet_v3_small_page_orientation(pretrained=False)
    custom_page_orientation_model.from_pretrained('<path_to_pt>')
    custom_crop_orientation_model = mobilenet_v3_small_crop_orientation(pretrained=False)
    custom_crop_orientation_model.from_pretrained('<path_to_pt>')

    predictor = ocr_predictor(
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
    )

    # Overwrite the default orientation models
    predictor.crop_orientation_predictor = crop_orientation_predictor(custom_crop_orientation_model)
    predictor.page_orientation_predictor = page_orientation_predictor(custom_page_orientation_model)

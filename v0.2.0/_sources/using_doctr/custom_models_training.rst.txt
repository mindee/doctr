Train your own model
====================

If the pretrained models don't meet your specific needs, you have the option to train your own model using the doctr library.
For details on the training process and the necessary data and data format, refer to the following links:

- `detection <https://github.com/mindee/doctr/tree/main/references/detection#readme>`_
- `recognition <https://github.com/mindee/doctr/tree/main/references/recognition#readme>`_

Loading your custom trained model
---------------------------------

This section shows how you can easily load a custom trained model in docTR.

.. tabs::

    .. tab:: TensorFlow

        .. code:: python3

            from doctr.models import ocr_predictor, db_resnet50, crnn_vgg16_bn

            # Load custom detection model
            det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
            det_model.load_weights("<path_to_checkpoint>/weights")
            predictor = ocr_predictor(det_arch=det_model, reco_arch="vitstr_small", pretrained=True)

            # Load custom recognition model
            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
            reco_model.load_weights("<path_to_checkpoint>/weights")
            predictor = ocr_predictor(det_arch="linknet_resnet18", reco_arch=reco_model, pretrained=True)

            # Load custom detection and recognition model
            det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
            det_model.load_weights("<path_to_checkpoint>/weights")
            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
            reco_model.load_weights("<path_to_checkpoint>/weights")
            predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model, pretrained=False)

    .. tab:: PyTorch

        .. code:: python3

            import torch
            from doctr.models import ocr_predictor, db_resnet50, crnn_vgg16_bn

            # Load custom detection model
            det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
            det_params = torch.load('<path_to_pt>', map_location="cpu")
            det_model.load_state_dict(det_params)
            predictor = ocr_predictor(det_arch=det_model, reco_arch="vitstr_small", pretrained=True)

            # Load custom recognition model
            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
            reco_params = torch.load('<path_to_pt>', map_location="cpu")
            reco_model.load_state_dict(reco_params)
            predictor = ocr_predictor(det_arch="linknet_resnet18", reco_arch=reco_model, pretrained=True)

            # Load custom detection and recognition model
            det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
            det_params = torch.load('<path_to_pt>', map_location="cpu")
            det_model.load_state_dict(det_params)
            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
            reco_params = torch.load('<path_to_pt>', map_location="cpu")
            reco_model.load_state_dict(reco_params)
            predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model, pretrained=False)

Load a custom recognition model trained on another vocabulary as the default one (French):

.. tabs::

    .. tab:: TensorFlow

        .. code:: python3

            from doctr.models import ocr_predictor, crnn_vgg16_bn
            from doctr.datasets import VOCABS

            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["german"])
            reco_model.load_weights("<path_to_checkpoint>/weights")

            predictor = ocr_predictor(det_arch='linknet_resnet18', reco_arch=reco_model, pretrained=True)

    .. tab:: PyTorch

        .. code:: python3

            import torch
            from doctr.models import ocr_predictor, crnn_vgg16_bn
            from doctr.datasets import VOCABS

            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=VOCABS["german"])
            reco_params = torch.load('<path_to_pt>', map_location="cpu")
            reco_model.load_state_dict(reco_params)

            predictor = ocr_predictor(det_arch='linknet_resnet18', reco_arch=reco_model, pretrained=True)

Load a custom trained KIE detection model:

.. tabs::

    .. tab:: TensorFlow

        .. code:: python3

            from doctr.models import kie_predictor, db_resnet50

            det_model = db_resnet50(pretrained=False, pretrained_backbone=False, class_names=['total', 'date'])
            det_model.load_weights("<path_to_checkpoint>/weights")
            kie_predictor(det_arch=det_model, reco_arch='crnn_vgg16_bn', pretrained=True)

    .. tab:: PyTorch

        .. code:: python3

            import torch
            from doctr.models import kie_predictor, db_resnet50

            det_model = db_resnet50(pretrained=False, pretrained_backbone=False, class_names=['total', 'date'])
            det_params = torch.load('<path_to_pt>', map_location="cpu")
            det_model.load_state_dict(det_params)
            kie_predictor(det_arch=det_model, reco_arch='crnn_vgg16_bn', pretrained=True)

Load a model with customized Preprocessor:

.. tabs::

    .. tab:: TensorFlow

        .. code:: python3

            from doctr.models.predictor import OCRPredictor
            from doctr.models.detection.predictor import DetectionPredictor
            from doctr.models.recognition.predictor import RecognitionPredictor
            from doctr.models.preprocessor import PreProcessor
            from doctr.models import db_resnet50, crnn_vgg16_bn

            det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
            det_model.load_weights("<path_to_checkpoint>/weights")
            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
            reco_model.load_weights("<path_to_checkpoint>/weights")

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

    .. tab:: PyTorch

        .. code:: python3

            import torch
            from doctr.models.predictor import OCRPredictor
            from doctr.models.detection.predictor import DetectionPredictor
            from doctr.models.recognition.predictor import RecognitionPredictor
            from doctr.models.preprocessor import PreProcessor
            from doctr.models import db_resnet50, crnn_vgg16_bn

            det_model = db_resnet50(pretrained=False, pretrained_backbone=False)
            det_params = torch.load('<path_to_pt>', map_location="cpu")
            det_model.load_state_dict(det_params)
            reco_model = crnn_vgg16_bn(pretrained=False, pretrained_backbone=False)
            reco_params = torch.load(<path_to_pt>, map_location="cpu")
            reco_model.load_state_dict(reco_params)

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

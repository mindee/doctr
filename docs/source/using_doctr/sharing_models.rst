Share your model with the community
===================================

docTR's focus is on open source, so if you also feel in love with than we appreciate sharing your trained model with the community.

.. currentmodule:: doctr.models.factory


Loading from Huggingface Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section shows you how you can easily load a pretrained model from the Huggingface Hub.

.. tabs::

    .. tab:: TensorFlow

        .. code:: python3

            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor, from_hub
            image = DocumentFile.from_images(['data/example.jpg'])
            # Load a custom detection model from huggingface hub
            det_model = from_hub('Felix92/doctr-dummy-tf-db-mobilenet-v3-large')
            # Load a custom recognition model from huggingface hub
            reco_model = from_hub('Felix92/doctr-dummy-tf-crnn-mobilenet-v3-large')
            # You can easily plug in this models to the OCR predictor
            predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
            result = predictor(image)

    .. tab:: PyTorch

        .. code:: python3

            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor, from_hub
            image = DocumentFile.from_images(['data/example.jpg'])
            # Load a custom detection model from huggingface hub
            det_model = from_hub('Felix92/doctr-dummy-torch-db-mobilenet-v3-large').eval()
            # Load a custom recognition model from huggingface hub
            reco_model = from_hub('Felix92/doctr-dummy-torch-crnn-mobilenet-v3-large').eval()
            # You can easily plug in this models to the OCR predictor
            predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
            result = predictor(image)


Pushing to the Huggingface Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also push your trained model to the Huggingface Hub.
You need only to provide the task type, a name for your trained model and the model name itself.

- Prerequisites:
    - huggingface account (you can easy create one at https://huggingface.co/)
    - installed Git LFS (check installation at: https://git-lfs.github.com/)

.. code:: python3

    from doctr import models
    my_awesome_model = models.recognition.crnn_mobilenet_v3_small(pretrained=True)
    push_to_hf_hub(my_awesome_model, model_name='doctr-crnn-mobilenet-v3-small', task='recognition', arch='crnn_mobilenet_v3_small')

It is also possible to push your model directly after training.

.. tabs::

    .. tab:: TensorFlow

        python3 ~/doctr/references/recognition/train_tensorflow.py crnn_mobilenet_v3_small --push-to-hub

    .. tab:: PyTorch

        python3 ~/doctr/references/recognition/train_pytorch.py crnn_mobilenet_v3_small --push-to-hub


Pretrained community models
---------------------------

This section is to provide some tables for pretrained community models.

Classification
^^^^^^^^^^^^^^

+---------------------------------+-------------------------------------+-----------------------+------------------------+
|        **Architecture**         |            **Repo_ID**              |     **Vocabulary**    |     **Framework**      |
+=================================+=====================================+=======================+========================+
| resnet18 (dummy)                | Felix92/doctr-dummy-torch-resnet18  | french                | PyTorch                |
+---------------------------------+-------------------------------------+-----------------------+------------------------+
| resnet18 (dummy)                | Felix92/doctr-dummy-tf-resnet18     | french                | TensorFlow             |
+---------------------------------+-------------------------------------+-----------------------+------------------------+


Detection
^^^^^^^^^

+---------------------------------+-------------------------------------------------+------------------------+
|        **Architecture**         |            **Repo_ID**                          |     **Framework**      |
+=================================+=================================================+========================+
| db_mobilenet_v3_large (dummy)   | Felix92/doctr-dummy-torch-db-mobilenet-v3-large | PyTorch                |
+---------------------------------+-------------------------------------------------+------------------------+
| db_mobilenet_v3_large (dummy)   | Felix92/doctr-dummy-tf-db-mobilenet-v3-large    | TensorFlow             |
+---------------------------------+-------------------------------------------------+------------------------+


Recognition
^^^^^^^^^^^

+---------------------------------+---------------------------------------------------+---------------------+------------------------+
|        **Architecture**         |            **Repo_ID**                            |     **Language**    |     **Framework**      |
+=================================+===================================================+=====================+========================+
| crnn_mobilenet_v3_large (dummy) | Felix92/doctr-dummy-torch-crnn-mobilenet-v3-large | french              | PyTorch                |
+---------------------------------+---------------------------------------------------+---------------------+------------------------+
| crnn_mobilenet_v3_large (dummy) | Felix92/doctr-dummy-tf-crnn-mobilenet-v3-large    | french              | TensorFlow             |
+---------------------------------+---------------------------------------------------+---------------------+------------------------+

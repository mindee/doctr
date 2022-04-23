Share your model with the community
===================================

A well-trained model is a good achievement but you might want to tune a few things to make it production-ready!

.. currentmodule:: doctr.models.factory


Loading from Huggingface Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section shows you how you can easily load a pretrained model from the huggingface hub.

    >>> from doctr.io import DocumentFile
    >>> from doctr.models import ocr_predictor, from_hub
    >>> image = DocumentFile.from_images(['data/example.jpg'])
    >>> # Load a custom detection model from huggingface hub
    >>> det_model = from_hub('Felix92/doctr-dummy-torch-db-mobilenet-v3-large')
    >>> # Load a custom recognition model from huggingface hub
    >>> reco_model = from_hub('Felix92/doctr-dummy-torch-crnn-mobilenet-v3-small')
    >>> # You can easily plug in this models to the OCR predictor
    >>> predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
    >>> result = predictor(image)


Pushing to the Huggingface Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Prerequisites: huggingface account / git lfs in repo installed / login to huggingface hub


Pretrained community models
---------------------------

Add some text / explain also that models directly after training can be pushed (references)

Classification
^^^^^^^^^^^^^^

+---------------------------------+-------------------------------------+-----------------------+------------------------+
|        **Architecture**         |            **Repo_ID**              |     **Vocabulary**    |     **Framework**      |
+=================================+=====================================+=======================+========================+
| resnet18 (dummy)                | Felix92/doctr-dummy-torch-resnet18  |french                 | PyTorch                |
+---------------------------------+-------------------------------------+-----------------------+------------------------+


Detection
^^^^^^^^^

+---------------------------------+-------------------------------------------------+------------------------+
|        **Architecture**         |            **Repo_ID**                          |     **Framework**      |
+=================================+=================================================+========================+
| db_mobilenet_v3_large (dummy)   | Felix92/doctr-dummy-torch-db-mobilenet-v3-large | PyTorch                |
+---------------------------------+-------------------------------------------------+------------------------+


Recognition
^^^^^^^^^^^

+---------------------------------+---------------------------------------------------+---------------------+-----------------------+
|        **Architecture**         |            **Repo_ID**                            |     **Language**    |     **Framework**     |
+=================================+===================================================+=====================+=======================+
| crnn_mobilenet_v3_small (dummy) | Felix92/doctr-dummy-torch-crnn-mobilenet-v3-small |french               |PyTorch                |
+---------------------------------+---------------------------------------------------+---------------------+-----------------------+

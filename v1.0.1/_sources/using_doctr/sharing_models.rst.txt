Share your model with the community
===================================

docTR's focus is on open source, so if you also feel in love with than we appreciate sharing your trained model with the community.
To make it easy for you, we have integrated a interface to the huggingface hub.

.. currentmodule:: doctr.models.factory


Loading from Huggingface Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section shows how you can easily load a pretrained model from the Huggingface Hub.

.. code:: python3

    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor, from_hub
    image = DocumentFile.from_images(['data/example.jpg'])
    # Load a custom detection model from huggingface hub
    det_model = from_hub('Felix92/doctr-torch-db-mobilenet-v3-large')
    # Load a custom recognition model from huggingface hub
    reco_model = from_hub('Felix92/doctr-torch-crnn-mobilenet-v3-large-french')
    # You can easily plug in this models to the OCR predictor
    predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
    result = predictor(image)


Pushing to the Huggingface Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also push your trained model to the Huggingface Hub.
You need only to provide the task type (classification, detection, recognition or obj_detection), a name for your trained model (NOTE:
existing repositories will not be overwritten) and the model name itself.

- Prerequisites:
    - Huggingface account (you can easy create one at https://huggingface.co/)
    - installed Git LFS (check installation at: https://git-lfs.github.com/) in the repository

.. code:: python3

    from doctr.models import recognition, login_to_hub, push_to_hf_hub
    login_to_hub()
    my_awesome_model = recognition.crnn_mobilenet_v3_large(pretrained=True)
    push_to_hf_hub(my_awesome_model, model_name='doctr-crnn-mobilenet-v3-large-french-v1', task='recognition', arch='crnn_mobilenet_v3_large')

It is also possible to push your model directly after training.

.. code:: bash

    python3 ~/doctr/references/recognition/train.py crnn_mobilenet_v3_large --name doctr-crnn-mobilenet-v3-large --push-to-hub


Pretrained community models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section is to provide some tables for pretrained community models.
Feel free to open a pull request or issue to add your model to this list.

Naming conventions
------------------

We suggest using the following naming conventions for your models:

**Classification:** ``doctr-<architecture>-<vocab>``

**Detection:** ``doctr-<architecture>``

**Recognition:** ``doctr-<architecture>-<vocab>``


Classification
--------------

+---------------------------------+-------------------------------------+-----------------------+
|        **Architecture**         |            **Repo_ID**              |     **Vocabulary**    |
+=================================+=====================================+=======================+
| resnet18 (dummy)                | Felix92/doctr-dummy-torch-resnet18  | french                |
+---------------------------------+-------------------------------------+-----------------------+


Detection
---------

+---------------------------------+-------------------------------------------------+------------------------+
|        **Architecture**         |            **Repo_ID**                          |     **Framework**      |
+=================================+=================================================+========================+
| db_resnet50                     | rania-sr/doctr-Detection-model-v1-arabic        | PyTorch                |
+---------------------------------+-------------------------------------------------+------------------------+


Recognition
-----------

+---------------------------------+---------------------------------------------------+---------------------+------------------------+
|        **Architecture**         |            **Repo_ID**                            |     **Language**    |     **Framework**      |
+=================================+===================================================+=====================+========================+
| crnn_vgg16_bn                   | tilman-rassy/doctr-crnn-vgg16-bn-fascan-v1        | french + german + § | PyTorch                |
+---------------------------------+---------------------------------------------------+---------------------+------------------------+
| parseq                          | Felix92/doctr-torch-parseq-multilingual-v1        | multilingual        | PyTorch                |
+---------------------------------+---------------------------------------------------+---------------------+------------------------+
| parseq                          | rania-sr/doctr-model-v1-arabic                    | arabic              | PyTorch                |
+---------------------------------+---------------------------------------------------+---------------------+------------------------+

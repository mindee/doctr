Integrate contributions into your pipeline
==========================================

The `contrib` module provides a collection of additional features which could be relevant for your document analysis pipeline.
The following sections will give you an overview of the available modules and features.

.. currentmodule:: doctr.contrib


Available contribution modules
------------------------------

**NOTE:** To use the contrib module, you need to install the `onnxruntime` package. You can install it using the following command:

.. code:: bash

    pip install python-doctr[contrib]
    # Or
    pip install onnxruntime  # pip install onnxruntime-gpu

Here are all contribution modules that are available through docTR:

ArtefactDetection
^^^^^^^^^^^^^^^^^

The ArtefactDetection module provides a set of functions to detect artefacts in the document images, such as logos, QR codes, bar codes, etc.
It is based on the YOLOv8 architecture, which is a state-of-the-art object detection model.

.. code:: python3

    from doctr.io import DocumentFile
    from doctr.contrib.artefacts import ArtefactDetection

    # Load the document
    doc = DocumentFile.from_images(["path/to/your/image"])
    detector = ArtefactDetection(batch_size=2, conf_threshold=0.5, iou_threshold=0.5)
    artefacts = detector(doc)

    # Visualize the detected artefacts
    detector.show()

You can also use your custom trained YOLOv8 model to detect artefacts or anything else you need.
Reference: `YOLOv8 <https://github.com/ultralytics/ultralytics>`_

**NOTE:** The YOLOv8 model (no Oriented Bounding Box (OBB) inference supported yet) needs to be provided as onnx exported model with a dynamic batch size.

.. code:: python3

    from doctr.contrib import ArtefactDetection

    detector = ArtefactDetection(model_path="path/to/your/model.onnx", labels=["table", "figure"])

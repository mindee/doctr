Text Detection
--------------

The task consists of localizing textual elements in a given image.
While those text elements can represent many things, in docTR, we will consider uninterrupted character sequences (words). Additionally, the localization can take several forms: from straight bounding boxes (delimited by the 2D coordinates of the top-left and bottom-right corner), to polygons, or binary segmentation (flagging which pixels belong to this element, and which don't).
Our latest detection models works with rotated and skewed documents!

Available architectures
^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

* `linknet_resnet18 <models.html#doctr.models.detection.linknet_resnet18>`_
* `db_resnet50 <models.html#doctr.models.detection.db_resnet50>`_
* `db_mobilenet_v3_large <models.html#doctr.models.detection.db_mobilenet_v3_large>`_

We also provide 2 models working with any kind of rotated documents:

* `linknet_resnet18_rotation <models.html#doctr.models.detection.linknet_resnet18_rotation>`_
* `db_resnet50_rotation <models.html#doctr.models.detection.db_resnet50_rotation>`_

For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


+------------------------------------------------------------------+----------------------------+----------------------------+---------+
|                                                                  |        FUNSD               |        CORD                |         |
+=================================+=================+==============+============+===============+============+===============+=========+
| **Architecture**                | **Input shape** | **# params** | **Recall** | **Precision** | **Recall** | **Precision** | **FPS** |
+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+
| db_resnet50                     | (1024, 1024, 3) | 25.2 M       | 82.14      | 87.64         | 92.49      | 89.66         | 2.1     |
+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+
| db_mobilenet_v3_large           | (1024, 1024, 3) |  4.2 M       | 79.35      | 84.03         | 81.14      | 66.85         |         |
+---------------------------------+-----------------+--------------+------------+---------------+------------+---------------+---------+


All text detection models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metrics being used are available in :ref:`metrics`.

*Disclaimer: both FUNSD subsets combined have 199 pages which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed after a warmup phase of 100 tensors (where the batch size is 1), by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `c5.x12large <https://aws.amazon.com/ec2/instance-types/c5/>`_ AWS instance (CPU Xeon Platinum 8275L).

Detection predictors
^^^^^^^^^^^^^^^^^^^^

`detection_predictor <models.html#doctr.models.detection.detection_predictor>`_ wraps your detection model to make it easily useable with your favorite deep learning framework seamlessly.

    >>> import numpy as np
    >>> from doctr.models import detection_predictor
    >>> predictor = detection_predictor('db_resnet50')
    >>> dummy_img = (255 * np.random.rand(800, 600, 3)).astype(np.uint8)
    >>> out = model([dummy_img])
You can pass specific boolean arguments to the predictor:

* `assume_straight_pages`: if you work with straight documents only, it will fit straight bounding boxes to the text areas.
* `preserve_aspect_ratio`: if you want to preserve the aspect ratio of your documents while resizing before sending them to the model.
* `symmetric_pad`: if you choose to preserve the aspect ratio, it will pad the image symmetrically and not from the bottom-right.

For instance, this snippet will instantiates a detection predictor able to detect text on rotated documents while preserving the aspect ratio:

    >>> from doctr.models import detection_predictor
    >>> predictor = detection_predictor('db_resnet50_rotation', pretrained=True, assume_straight_pages=False, preserve_aspect_ratio=True)
NB: for the moment, `db_resnet50_rotation` is pretrained in Pytorch only and `linknet_resnet18_rotation` in Tensorflow only.

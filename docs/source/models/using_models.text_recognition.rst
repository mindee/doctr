Text Recognition
----------------

The task consists of transcribing the character sequence in a given image.


Available architectures
^^^^^^^^^^^^^^^^^^^^^^^

The following architectures are currently supported:

* `crnn_vgg16_bn <models.html#doctr.models.recognition.crnn_vgg16_bn>`_
* `crnn_mobilenet_v3_small <models.html#doctr.models.recognition.crnn_mobilenet_v3_small>`_
* `crnn_mobilenet_v3_large <models.html#doctr.models.recognition.crnn_mobilenet_v3_large>`_
* `sar_resnet31 <models.html#doctr.models.recognition.sar_resnet31>`_
* `master <models.html#doctr.models.recognition.master>`_


For a comprehensive comparison, we have compiled a detailed benchmark on publicly available datasets:


.. list-table:: Text recognition model zoo
   :header-rows: 1

   * - Architecture
     - Input shape
     - # params
     - FUNSD
     - CORD
     - FPS
   * - crnn_vgg16_bn
     - (32, 128, 3)
     - 15.8M
     - 87.18
     - 92.93
     - 12.8
   * - crnn_mobilenet_v3_small
     - (32, 128, 3)
     - 2.1M
     - 86.21
     - 90.56
     -
   * - crnn_mobilenet_v3_large
     - (32, 128, 3)
     - 4.5M
     - 86.95
     - 92.03
     -
   * - sar_resnet31
     - (32, 128, 3)
     - 56.2M
     - **87.70**
     - **93.41**
     - 2.7
   * - master
     - (32, 128, 3)
     - 67.7M
     - 87.62
     - 93.27
     -

All text recognition models above have been evaluated using both the training and evaluation sets of FUNSD and CORD (cf. :ref:`datasets`).
Explanations about the metric being used (exact match) are available in :ref:`metrics`.

While most of our recognition models were trained on our french vocab (cf. :ref:`vocabs`), you can easily access the vocab of any model as follows:

    >>> from doctr.models import recognition_predictor
    >>> predictor = recognition_predictor('crnn_vgg16_bn')
    >>> print(predictor.model.cfg['vocab'])

*Disclaimer: both FUNSD subsets combine have 30595 word-level crops which might not be representative enough of the model capabilities*

FPS (Frames per second) is computed after a warmup phase of 100 tensors (where the batch size is 1), by measuring the average number of processed tensors per second over 1000 samples. Those results were obtained on a `c5.x12large <https://aws.amazon.com/ec2/instance-types/c5/>`_ AWS instance (CPU Xeon Platinum 8275L).


Recognition predictors
^^^^^^^^^^^^^^^^^^^^^^
`recognition_predictor <models.html#doctr.models.recognition.recognition_predictor>`_ wraps your recognition model to make it easily useable with your favorite deep learning framework seamlessly.

    >>> import numpy as np
    >>> from doctr.models import recognition_predictor
    >>> predictor = recognition_predictor('crnn_vgg16_bn')
    >>> dummy_img = (255 * np.random.rand(50, 150, 3)).astype(np.uint8)
    >>> out = model([dummy_img])

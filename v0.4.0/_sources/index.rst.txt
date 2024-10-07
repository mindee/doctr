DocTR: Document Text Recognition
================================

State-of-the-art Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 (PyTorch now in beta)

.. image:: https://github.com/mindee/doctr/releases/download/v0.2.0/ocr.png
        :align: center


DocTR provides an easy and powerful way to extract valuable information from your documents:

* |:receipt:| **for automation**: seemlessly process documents for Natural Language Understanding tasks: we provide OCR predictors to parse textual information (localize and identify each word) from your documents.
* |:woman_scientist:| **for research**: quickly compare your own architectures speed & performances with state-of-art models on public datasets.

Welcome to the documentation of `DocTR <https://github.com/mindee/doctr>`_!



Main Features
-------------

* |:robot:| Robust 2-stage (detection + recognition) OCR predictors with pretrained parameters
* |:zap:| User-friendly, 3 lines of code to load a document and extract text with a predictor
* |:rocket:| State-of-the-art performances on public document datasets, comparable with GoogleVision/AWS Textract
* |:zap:| Optimized for inference speed on both CPU & GPU
* |:bird:| Light package, small dependencies
* |:tools:| Daily maintained
* |:factory:| Easy integration


Getting Started
---------------

.. toctree::
   :maxdepth: 2

   installing


Build & train your predictor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Compose your own end-to-end OCR predictor: mix and match detection & recognition predictors (all-pretrained)
* Fine-tune or train from scratch any detection or recognition model to specialize on your data


Model zoo
^^^^^^^^^

Text detection models
"""""""""""""""""""""
   * `DBNet <https://arxiv.org/pdf/1911.08947.pdf>`_ (Differentiable Binarization)
   * `LinkNet <https://arxiv.org/pdf/1707.03718.pdf>`_

Text recognition models
"""""""""""""""""""""""
   * `SAR <https://arxiv.org/pdf/1811.00751.pdf>`_ (Show, Attend and Read)
   * `CRNN <https://arxiv.org/pdf/1507.05717.pdf>`_ (Convolutional Recurrent Neural Network)
   * `MASTER <https://arxiv.org/pdf/1910.02562.pdf>`_ (Multi-Aspect Non-local Network for Scene Text Recognition)


Supported datasets
^^^^^^^^^^^^^^^^^^
   * FUNSD from `"FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents" <https://arxiv.org/pdf/1905.13538.pdf>`_.
   * CORD from `"CORD: A Consolidated Receipt Dataset forPost-OCR Parsing" <https://openreview.net/pdf?id=SJl3z659UH>`_.
   * SROIE from `ICDAR 2019 <https://rrc.cvc.uab.es/?ch=13>`_.


.. toctree::
   :maxdepth: 2
   :caption: Notes

   changelog


.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   datasets
   documents
   models
   transforms
   utils

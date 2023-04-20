docTR: Document Text Recognition
================================

State-of-the-art Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 & PyTorch

.. image:: https://github.com/mindee/doctr/releases/download/v0.2.0/ocr.png
        :align: center


DocTR provides an easy and powerful way to extract valuable information from your documents:

* |:receipt:| **for automation**: seemlessly process documents for Natural Language Understanding tasks: we provide OCR predictors to parse textual information (localize and identify each word) from your documents.
* |:woman_scientist:| **for research**: quickly compare your own architectures speed & performances with state-of-art models on public datasets.


Main Features
-------------

* |:robot:| Robust 2-stage (detection + recognition) OCR predictors with pretrained parameters
* |:zap:| User-friendly, 3 lines of code to load a document and extract text with a predictor
* |:rocket:| State-of-the-art performances on public document datasets, comparable with GoogleVision/AWS Textract
* |:zap:| Optimized for inference speed on both CPU & GPU
* |:bird:| Light package, minimal dependencies
* |:tools:| Actively maintained by Mindee
* |:factory:| Easy integration (available templates for browser demo & API deployment)


.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installing
   notebooks


Model zoo
^^^^^^^^^

Text detection models
"""""""""""""""""""""
   * DBNet from `"Real-time Scene Text Detection with Differentiable Binarization" <https://arxiv.org/pdf/1911.08947.pdf>`_
   * LinkNet from `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation" <https://arxiv.org/pdf/1707.03718.pdf>`_

Text recognition models
"""""""""""""""""""""""
   * SAR from `"Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_
   * CRNN from `"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_
   * MASTER from `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition" <https://arxiv.org/pdf/1910.02562.pdf>`_


Supported datasets
^^^^^^^^^^^^^^^^^^
   * FUNSD from `"FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents" <https://arxiv.org/pdf/1905.13538.pdf>`_.
   * CORD from `"CORD: A Consolidated Receipt Dataset forPost-OCR Parsing" <https://openreview.net/pdf?id=SJl3z659UH>`_.
   * SROIE from `ICDAR 2019 <https://rrc.cvc.uab.es/?ch=13>`_.


.. toctree::
   :maxdepth: 2
   :caption: Using docTR
   :hidden:

   using_models
   using_model_export


.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :hidden:

   datasets
   io
   models
   transforms
   utils


.. toctree::
   :maxdepth: 2
   :caption: Notes
   :hidden:

   changelog

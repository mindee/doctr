DocTR: Document Text Recognition
================================

State-of-the-art Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2

DocTR provides an easy and powerful way to extract valuable information from your documents:

* |:receipt:| **for automation**: seemlessly process documents for Natural Language Understanding tasks: we provide OCR predictors to parse textual information (localize and identify each word) from your documents.
* |:woman_scientist:| **for research**: quickly compare your own architectures speed & performances with state-of-art models on public datasets.

This is the documentation of our repository `doctr <https://github.com/mindee/doctr>`_.


Features
--------

* |:robot:| Robust 2-stages (detection + recognition) OCR predictors fully trained
* |:zap:| User-friendly, 3 lines of code to load a document and extract text with a predictor
* |:rocket:| State-of-the-art performances on public document datasets, comparable with GoogleVision/AWS Textract
* |:zap:| Predictors optimized to be very fast on both CPU & GPU
* |:bird:| Light package, small dependencies
* |:tools:| Daily maintained
* |:factory:| Easily integrable


|:scientist:| Build & train your predictor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* |:construction_worker:| Compose your own end-to-end OCR predictor: mix and match detection & recognition predictors (all-pretrained)
* |:construction_worker:| Fine-tune or train from scratch any detection or recognition model to specialize on your data


|:toolbox:| Implemented models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Detection models
""""""""""""""""
   * DB (Differentiable Binarization), `"Real-time Scene Text Detection with Differentiable Binarization" <https://arxiv.org/pdf/1911.08947.pdf>`_.
   * LinkNet, `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation" <https://arxiv.org/pdf/1707.03718.pdf>`_.

Recognition models
""""""""""""""""""
   * SAR (Show, Attend and Read), `"Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.
   * CRNN (Convolutional Recurrent Neural Network), `"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.


|:receipt:| Integrated datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   * FUNSD from `"FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents" <https://arxiv.org/pdf/1905.13538.pdf>`_.
   * CORD from `"CORD: A Consolidated Receipt Dataset forPost-OCR Parsing" <https://openreview.net/pdf?id=SJl3z659UH>`_.


Getting Started
---------------

.. toctree::
   :maxdepth: 2

   installing


Contents
--------

.. toctree::
   :maxdepth: 1

   datasets
   documents
   models
   transforms
   utils


.. automodule:: doctr
   :members:

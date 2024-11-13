********************************
docTR: Document Text Recognition
********************************

State-of-the-art Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 & PyTorch

.. image:: https://github.com/mindee/doctr/releases/download/v0.2.0/ocr.png
        :align: center


DocTR provides an easy and powerful way to extract valuable information from your documents:

* |:receipt:| **for automation**: seamlessly process documents for Natural Language Understanding tasks: we provide OCR predictors to parse textual information (localize and identify each word) from your documents.
* |:woman_scientist:| **for research**: quickly compare your own architectures speed & performances with state-of-art models on public datasets.


Main Features
-------------

* |:robot:| Robust 2-stage (detection + recognition) OCR predictors with pretrained parameters
* |:zap:| User-friendly, 3 lines of code to load a document and extract text with a predictor
* |:rocket:| State-of-the-art performance on public document datasets, comparable with GoogleVision/AWS Textract
* |:zap:| Optimized for inference speed on both CPU & GPU
* |:bird:| Light package, minimal dependencies
* |:tools:| Actively maintained by Mindee
* |:factory:| Easy integration (available templates for browser demo & API deployment)


.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   getting_started/installing
   notebooks


Model zoo
^^^^^^^^^

Text detection models
"""""""""""""""""""""
* DBNet from `"Real-time Scene Text Detection with Differentiable Binarization" <https://arxiv.org/pdf/1911.08947.pdf>`_
* LinkNet from `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation" <https://arxiv.org/pdf/1707.03718.pdf>`_
* FAST from `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation" <https://arxiv.org/pdf/2111.02394.pdf>`_

Text recognition models
"""""""""""""""""""""""
* SAR from `"Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_
* CRNN from `"An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_
* MASTER from `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition" <https://arxiv.org/pdf/1910.02562.pdf>`_
* ViTSTR from `"Vision Transformer for Fast and Efficient Scene Text Recognition" <https://arxiv.org/pdf/2105.08582.pdf>`_
* PARSeq from `"Scene Text Recognition with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_


Supported datasets
^^^^^^^^^^^^^^^^^^
* FUNSD from `"FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents" <https://arxiv.org/pdf/1905.13538.pdf>`_.
* CORD from `"CORD: A Consolidated Receipt Dataset forPost-OCR Parsing" <https://openreview.net/pdf?id=SJl3z659UH>`_.
* SROIE from `ICDAR 2019 <https://rrc.cvc.uab.es/?ch=13>`_.
* IIIT-5k from `CVIT <https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset>`_.
* Street View Text from `"End-to-End Scene Text Recognition" <http://vision.ucsd.edu/~kai/pubs/wang_iccv2011.pdf>`_.
* SynthText from `Visual Geometry Group <https://www.robots.ox.ac.uk/~vgg/data/scenetext/>`_.
* SVHN from `"Reading Digits in Natural Images with Unsupervised Feature Learning" <http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf>`_.
* IC03 from `ICDAR 2003 <http://www.iapr-tc11.org/mediawiki/index.php?title=ICDAR_2003_Robust_Reading_Competitions>`_.
* IC13 from `ICDAR 2013 <http://dagdata.cvc.uab.es/icdar2013competition/>`_.
* IMGUR5K from `"TextStyleBrush: Transfer of Text Aesthetics from a Single Example" <https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset>`_.
* MJSynth from `"Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition" <https://www.robots.ox.ac.uk/~vgg/data/text/>`_.
* IIITHWS from `"Generating Synthetic Data for Text Recognition" <https://github.com/kris314/hwnet>`_.
* WILDRECEIPT from `"Spatial Dual-Modality Graph Reasoning for Key Information Extraction" <https://arxiv.org/pdf/2103.14470v1.pdf>`_.


.. toctree::
   :maxdepth: 2
   :caption: Using docTR
   :hidden:

   using_doctr/using_models
   using_doctr/using_datasets
   using_doctr/using_contrib_modules
   using_doctr/sharing_models
   using_doctr/using_model_export
   using_doctr/custom_models_training
   using_doctr/running_on_aws


.. toctree::
   :maxdepth: 2
   :caption: Community
   :hidden:

   community/resources


.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :hidden:

   modules/contrib
   modules/datasets
   modules/io
   modules/models
   modules/transforms
   modules/utils


.. toctree::
   :maxdepth: 2
   :caption: Contributing
   :hidden:

   contributing/code_of_conduct
   contributing/contributing


.. toctree::
   :maxdepth: 2
   :caption: Notes
   :hidden:

   changelog

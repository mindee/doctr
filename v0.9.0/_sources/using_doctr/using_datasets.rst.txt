Choose a ready to use dataset
=============================

Whether it is for training or for evaluation, having predefined objects to access datasets in your prefered framework
can be a significant save of time.

.. currentmodule:: doctr.datasets


Available Datasets
------------------
In the package reference you will also find some samples for each dataset.

Here are all datasets that are available through docTR:

Detection
^^^^^^^^^

This datasets contains the information to train or validate a text detection model.

+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
|        **Dataset**          |        **Train Samples**        |        **Test Samples**         |       **Information**            |
+=============================+=================================+=================================+==================================+
| FUNSD                       | 149                             | 50                              |                                  |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| SROIE                       | 626                             | 360                             |                                  |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| CORD                        | 800                             | 100                             |                                  |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| IIIT5K                      | 2000                            | 3000                            |                                  |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| SVT                         | 100                             | 249                             |                                  |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| SVHN                        | 33402                           | 13068                           | Character Localization           |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| SynthText                   | 772875                          | 85875                           |                                  |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| IC03                        | 246                             | 249                             |                                  |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| IC13                        | 229                             | 233                             | external resources               |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| IMGUR5K                     | 7149                            | 796                             | Handwritten / external resources |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+
| WILDRECEIPT                 | 1268                            | 472                             | external resources               |
+-----------------------------+---------------------------------+---------------------------------+----------------------------------+

.. code:: python3

    from doctr.datasets import CORD
    # Load straight boxes
    train_set = CORD(train=True, download=True, detection_task=True)
    # Load rotated boxes
    train_set = CORD(train=True, download=True, use_polygons=True, detection_task=True)
    img, target = train_set[0]


Recognition
^^^^^^^^^^^

This datasets contains the information to train or validate a text recognition model.

+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
|        **Dataset**          |        **Train Samples**        |        **Test Samples**         |               **Information**               |
+=============================+=================================+=================================+=============================================+
| FUNSD                       | 21888                           | 8707                            | english                                     |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| SROIE                       | 33608                           | 19342                           | english / only uppercase labels             |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| CORD                        | 19370                           | 2186                            | english                                     |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| IIIT5K                      | 2000                            | 3000                            | english                                     |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| SVT                         | 257                             | 647                             | english / only uppercase labels             |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| SVHN                        | 73257                           | 26032                           | digits                                      |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| SynthText                   | ~7100000                        | 707470                          | english                                     |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| IC03                        | 1156                            | 1107                            | english                                     |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| IC13                        | 849                             | 1095                            | english / external resources                |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| IMGUR5K                     | 207901                          | 22672                           | english / handwritten / external resources  |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| MJSynth                     | 7581382                         | 1337891                         | english / external resources                |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| IIITHWS                     | 7141797                         | 793533                          | english / handwritten / external resources  |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+
| WILDRECEIPT                 | 49377                           | 19598                           | english / external resources                |
+-----------------------------+---------------------------------+---------------------------------+---------------------------------------------+

.. code:: python3

    from doctr.datasets import CORD
    # Crop boxes as is (can contain irregular)
    train_set = CORD(train=True, download=True, recognition_task=True)
    # Crop rotated boxes (always regular)
    train_set = CORD(train=True, download=True, use_polygons=True, recognition_task=True)
    img, target = train_set[0]


OCR
^^^

The same dataset table as for detection, but with information about the bounding boxes and labels.

.. code:: python3

    from doctr.datasets import CORD
    # Load straight boxes
    train_set = CORD(train=True, download=True)
    # Load rotated boxes
    train_set = CORD(train=True, download=True, use_polygons=True)
    img, target = train_set[0]


Object Detection
^^^^^^^^^^^^^^^^

This datasets contains the information to train or validate a object detection model.

+-----------------------------+---------------------------------+---------------------------------+-------------------------------------------------------+
|        **Dataset**          |        **Train Samples**        |        **Test Samples**         |                   **Information**                     |
+=============================+=================================+=================================+=======================================================+
| DocArtefacts                | 2700                            | 300                             |["background", "qr_code", "bar_code", "logo", "photo"] |
+-----------------------------+---------------------------------+---------------------------------+-------------------------------------------------------+

.. code:: python3

    from doctr.datasets import DocArtefacts
    train_set = DocArtefacts(train=True, download=True)
    img, target = train_set[0]


Synthetic dataset generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

docTR provides also some generator objects, which can be used to generate synthetic datasets.
Both are also integrated in the training scripts to train a classification or recognition model.

.. code:: python3

    from doctr.datasets import CharacterGenerator
    ds = CharacterGenerator(vocab='abdef', num_samples=100)
    img, target = ds[0]

.. code:: python3

    from doctr.datasets import WordGenerator
    ds = WordGenerator(vocab='abdef', min_chars=1, max_chars=32, num_samples=100)
    img, target = ds[0]


Use your own datasets
---------------------

Since many documents include sensitive / personal information, we are not able to share all the data that has been used for this project.
However, we provide some guidance on how to format your own dataset into the same format so that you can use all docTR tools more easily.
You can find further information about the format in references.

.. code:: python3

    from doctr.datasets import DetectionDataset
    # Load a detection dataset
    train_set = DetectionDataset(img_folder="/path/to/images", label_path="/path/to/labels.json")
    # Load a recognition Dataset
    train_set = RecognitionDataset(img_folder="/path/to/images", labels_path="/path/to/labels.json")
    # Load a OCR dataset which contains anotations for the boxes and labels
    train_set = OCRDataset(img_folder="/path/to/images", label_file="/path/to/labels.json")
    img, target = train_set[0]


Data Loading
------------

Each dataset has its specific way to load a sample, but handling batch aggregation and the underlying iterator is a task deferred to another object in docTR.

.. code:: python3

    from doctr.datasets import CORD, DataLoader
    train_set = CORD(train=True, download=True)
    train_loader = DataLoader(train_set, batch_size=32)
    train_iter = iter(train_loader)
    images, targets = next(train_iter)

Choosing the right model
========================

The full Optical Character Recognition task can be seen as two consecutive tasks: text detection and text recognition.
Either performed at once or separately, to each task corresponds a type of deep learning architecture.

.. currentmodule:: doctr.models

For a given task, docTR provides a Predictor, which is composed of 2 components:

* PreProcessor: a module in charge of making inputs directly usable by the deep learning model.
* Model: a deep learning model, implemented with all supported deep learning backends (TensorFlow & PyTorch) along with its specific post-processor to make outputs structured and reusable.

.. toctree::
   :maxdepth: 3

   using_models.text_detection
   using_models.text_recognition
   using_models.end_to_end

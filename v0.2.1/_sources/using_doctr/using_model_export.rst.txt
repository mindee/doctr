Preparing your model for inference
==================================

A well-trained model is a good achievement but you might want to tune a few things to make it production-ready!

.. currentmodule:: doctr.models.export


Model compression
-----------------

This section is meant to help you perform inference with compressed versions of your model.


TensorFlow Lite
^^^^^^^^^^^^^^^

TensorFlow provides utilities packaged as TensorFlow Lite to take resource constraints into account. You can easily convert any Keras model into a serialized TFLite version as follows:

    >>> import tensorflow as tf
    >>> from tensorflow.keras import Sequential
    >>> from doctr.models import conv_sequence
    >>> model = Sequential(conv_sequence(32, 'relu', True, kernel_size=3, input_shape=(224, 224, 3)))
    >>> converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    >>> serialized_model = converter.convert()

Half-precision
^^^^^^^^^^^^^^

If you want to convert it to half-precision using your TFLite converter

    >>> converter.optimizations = [tf.lite.Optimize.DEFAULT]
    >>> converter.target_spec.supported_types = [tf.float16]
    >>> serialized_model = converter.convert()


Post-training quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally if you wish to quantize the model with your TFLite converter

    >>> converter.optimizations = [tf.lite.Optimize.DEFAULT]
    >>> # Float fallback for operators that do not have an integer implementation
    >>> def representative_dataset():
    >>>     for _ in range(100): yield [np.random.rand(1, *input_shape).astype(np.float32)]
    >>> converter.representative_dataset = representative_dataset
    >>> converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    >>> converter.inference_input_type = tf.int8
    >>> converter.inference_output_type = tf.int8
    >>> serialized_model = converter.convert()


Using SavedModel
----------------

Additionally, models in docTR inherit TensorFlow 2 model properties and can be exported to
`SavedModel <https://www.tensorflow.org/guide/saved_model>`_ format as follows:


    >>> import tensorflow as tf
    >>> from doctr.models import db_resnet50
    >>> model = db_resnet50(pretrained=True)
    >>> input_t = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> _ = model(input_t, training=False)
    >>> tf.saved_model.save(model, 'path/to/your/folder/db_resnet50/')

And loaded just as easily:


    >>> import tensorflow as tf
    >>> model = tf.saved_model.load('path/to/your/folder/db_resnet50/')

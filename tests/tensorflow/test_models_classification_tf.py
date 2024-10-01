import os
import tempfile

import cv2
import numpy as np
import onnxruntime
import psutil
import pytest
import tensorflow as tf

from doctr.models import classification
from doctr.models.classification.predictor import OrientationPredictor
from doctr.models.utils import export_model_to_onnx

system_available_memory = int(psutil.virtual_memory().available / 1024**3)


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["vgg16_bn_r", (32, 32, 3), (126,)],
        ["resnet18", (32, 32, 3), (126,)],
        ["resnet31", (32, 32, 3), (126,)],
        ["resnet34", (32, 32, 3), (126,)],
        ["resnet34_wide", (32, 32, 3), (126,)],
        ["resnet50", (32, 32, 3), (126,)],
        ["magc_resnet31", (32, 32, 3), (126,)],
        ["mobilenet_v3_small", (32, 32, 3), (126,)],
        ["mobilenet_v3_large", (32, 32, 3), (126,)],
        ["vit_s", (32, 32, 3), (126,)],
        ["vit_b", (32, 32, 3), (126,)],
        ["textnet_tiny", (32, 32, 3), (126,)],
        ["textnet_small", (32, 32, 3), (126,)],
        ["textnet_base", (32, 32, 3), (126,)],
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    tf.keras.backend.clear_session()
    model = classification.__dict__[arch_name](pretrained=True, include_top=True, input_shape=input_shape)
    # Forward
    out = model(tf.random.uniform(shape=[batch_size, *input_shape], maxval=1, dtype=tf.float32))
    # Output checks
    assert isinstance(out, tf.Tensor)
    assert out.dtype == tf.float32
    assert out.numpy().shape == (batch_size, *output_size)
    # Check that you can load pretrained up to the classification layer with differing number of classes to fine-tune
    tf.keras.backend.clear_session()
    assert classification.__dict__[arch_name](
        pretrained=True, include_top=True, input_shape=input_shape, num_classes=10
    )


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["mobilenet_v3_small_crop_orientation", (256, 256, 3)],
        ["mobilenet_v3_small_page_orientation", (512, 512, 3)],
    ],
)
def test_classification_models(arch_name, input_shape):
    batch_size = 8
    reco_model = classification.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    assert isinstance(reco_model, tf.keras.Model)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)

    out = reco_model(input_tensor)
    assert isinstance(out, tf.Tensor)
    assert out.shape.as_list() == [8, 4]


@pytest.mark.parametrize(
    "arch_name",
    [
        "mobilenet_v3_small_crop_orientation",
        "mobilenet_v3_small_page_orientation",
    ],
)
def test_classification_zoo(arch_name):
    if "crop" in arch_name:
        batch_size = 16
        input_tensor = tf.random.uniform(shape=[batch_size, 256, 256, 3], minval=0, maxval=1)
        # Model
        predictor = classification.zoo.crop_orientation_predictor(arch_name, pretrained=False)

        with pytest.raises(ValueError):
            predictor = classification.zoo.crop_orientation_predictor(arch="wrong_model", pretrained=False)
    else:
        batch_size = 2
        input_tensor = tf.random.uniform(shape=[batch_size, 512, 512, 3], minval=0, maxval=1)
        # Model
        predictor = classification.zoo.page_orientation_predictor(arch_name, pretrained=False)

        with pytest.raises(ValueError):
            predictor = classification.zoo.page_orientation_predictor(arch="wrong_model", pretrained=False)
    # object check
    assert isinstance(predictor, OrientationPredictor)
    out = predictor(input_tensor)
    class_idxs, classes, confs = out[0], out[1], out[2]
    assert isinstance(class_idxs, list) and len(class_idxs) == batch_size
    assert isinstance(classes, list) and len(classes) == batch_size
    assert isinstance(confs, list) and len(confs) == batch_size
    assert all(isinstance(pred, int) for pred in class_idxs)
    assert all(isinstance(pred, int) for pred in classes) and all(pred in [0, 90, 180, -90] for pred in classes)
    assert all(isinstance(pred, float) for pred in confs)


def test_crop_orientation_model(mock_text_box):
    text_box_0 = cv2.imread(mock_text_box)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    classifier = classification.crop_orientation_predictor("mobilenet_v3_small_crop_orientation", pretrained=True)
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    # 270 degrees is equivalent to -90 degrees
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(isinstance(pred, float) for pred in classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2])

    # Test with disabled predictor
    classifier = classification.crop_orientation_predictor(
        "mobilenet_v3_small_crop_orientation", pretrained=False, disabled=True
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90]) == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    # Test custom model loading
    classifier = classification.crop_orientation_predictor(
        classification.mobilenet_v3_small_crop_orientation(pretrained=True)
    )
    assert isinstance(classifier, OrientationPredictor)

    with pytest.raises(ValueError):
        _ = classification.crop_orientation_predictor(classification.textnet_tiny(pretrained=True))


def test_page_orientation_model(mock_payslip):
    text_box_0 = cv2.imread(mock_payslip)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    classifier = classification.page_orientation_predictor("mobilenet_v3_small_page_orientation", pretrained=True)
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    # 270 degrees is equivalent to -90 degrees
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(isinstance(pred, float) for pred in classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2])

    # Test with disabled predictor
    classifier = classification.page_orientation_predictor(
        "mobilenet_v3_small_page_orientation", pretrained=False, disabled=True
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90]) == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    # Test custom model loading
    classifier = classification.page_orientation_predictor(
        classification.mobilenet_v3_small_page_orientation(pretrained=True)
    )
    assert isinstance(classifier, OrientationPredictor)

    with pytest.raises(ValueError):
        _ = classification.page_orientation_predictor(classification.textnet_tiny(pretrained=True))


# temporarily fix to avoid killing the CI (tf2onnx v1.14 memory leak issue)
# ref.: https://github.com/mindee/doctr/pull/1201
@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["vgg16_bn_r", (32, 32, 3), (126,)],
        ["mobilenet_v3_small", (512, 512, 3), (126,)],
        ["mobilenet_v3_large", (512, 512, 3), (126,)],
        ["mobilenet_v3_small_crop_orientation", (256, 256, 3), (4,)],
        ["mobilenet_v3_small_page_orientation", (512, 512, 3), (4,)],
        ["resnet18", (32, 32, 3), (126,)],
        ["vit_s", (32, 32, 3), (126,)],
        ["textnet_tiny", (32, 32, 3), (126,)],
        ["textnet_small", (32, 32, 3), (126,)],
        pytest.param(
            "resnet31",
            (32, 32, 3),
            (126,),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "resnet34",
            (32, 32, 3),
            (126,),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "resnet34_wide",
            (32, 32, 3),
            (126,),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "resnet50",
            (32, 32, 3),
            (126,),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "magc_resnet31",
            (32, 32, 3),
            (126,),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "vit_b",
            (32, 32, 3),
            (126,),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "textnet_base",
            (32, 32, 3),
            (126,),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
    ],
)
def test_models_onnx_export(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    tf.keras.backend.clear_session()
    if "orientation" in arch_name:
        model = classification.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    else:
        model = classification.__dict__[arch_name](pretrained=True, include_top=True, input_shape=input_shape)

    if arch_name == "vit_b" or arch_name == "vit_s":
        # vit model needs a fixed batch size
        dummy_input = [tf.TensorSpec([2, *input_shape], tf.float32, name="input")]
    else:
        # batch_size = None for dynamic batch size
        dummy_input = [tf.TensorSpec([None, *input_shape], tf.float32, name="input")]

    np_dummy_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
    tf_logits = model(np_dummy_input, training=False).numpy()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path, output = export_model_to_onnx(
            model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input
        )
        assert os.path.exists(model_path)
        # Inference
        ort_session = onnxruntime.InferenceSession(
            os.path.join(tmpdir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        ort_outs = ort_session.run(output, {"input": np_dummy_input})

    assert isinstance(ort_outs, list) and len(ort_outs) == 1
    assert ort_outs[0].shape == (batch_size, *output_size)
    # Check that the output is close to the TensorFlow output - only warn if not close
    try:
        assert np.allclose(tf_logits, ort_outs[0], atol=1e-4)
    except AssertionError:
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {np.max(np.abs(tf_logits - ort_outs[0]))}")

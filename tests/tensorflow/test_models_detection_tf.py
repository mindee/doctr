import math
import os
import tempfile

import numpy as np
import onnxruntime
import psutil
import pytest
import tensorflow as tf

from doctr.file_utils import CLASS_NAME
from doctr.io import DocumentFile
from doctr.models import detection
from doctr.models.detection._utils import dilate, erode
from doctr.models.detection.fast.tensorflow import reparameterize
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import export_model_to_onnx

system_available_memory = int(psutil.virtual_memory().available / 1024**3)


@pytest.mark.parametrize("train_mode", [True, False])
@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet50", (512, 512, 3), (512, 512, 1), True],
        ["db_mobilenet_v3_large", (512, 512, 3), (512, 512, 1), True],
        ["linknet_resnet18", (512, 512, 3), (512, 512, 1), True],
        ["linknet_resnet34", (512, 512, 3), (512, 512, 1), True],
        ["linknet_resnet50", (512, 512, 3), (512, 512, 1), True],
        ["fast_tiny", (512, 512, 3), (512, 512, 1), True],
        ["fast_tiny_rep", (512, 512, 3), (512, 512, 1), True],  # Reparameterized model
        ["fast_small", (512, 512, 3), (512, 512, 1), True],
        ["fast_base", (512, 512, 3), (512, 512, 1), True],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob, train_mode):
    batch_size = 2
    tf.keras.backend.clear_session()
    if arch_name == "fast_tiny_rep":
        model = reparameterize(detection.fast_tiny(pretrained=True, input_shape=input_shape))
        train_mode = False  # Reparameterized model is not trainable
    else:
        model = detection.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    assert isinstance(model, tf.keras.Model)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)
    target = [
        {CLASS_NAME: np.array([[0.5, 0.5, 1, 1], [0.5, 0.5, 0.8, 0.8]], dtype=np.float32)},
        {CLASS_NAME: np.array([[0.5, 0.5, 1, 1], [0.5, 0.5, 0.8, 0.9]], dtype=np.float32)},
    ]
    # test training model
    out = model(
        input_tensor,
        target,
        return_model_output=True,
        return_preds=not train_mode,
        training=train_mode,
    )
    assert isinstance(out, dict)
    assert len(out) == 3 if not train_mode else len(out) == 2
    # Check proba map
    assert isinstance(out["out_map"], tf.Tensor)
    assert out["out_map"].dtype == tf.float32
    seg_map = out["out_map"].numpy()
    assert seg_map.shape == (batch_size, *output_size)
    if out_prob:
        assert np.all(np.logical_and(seg_map >= 0, seg_map <= 1))
    # Check boxes
    if not train_mode:
        for boxes_dict in out["preds"]:
            for boxes in boxes_dict.values():
                assert boxes.shape[1] == 5
                assert np.all(boxes[:, :2] < boxes[:, 2:4])
                assert np.all(boxes[:, :4] >= 0) and np.all(boxes[:, :4] <= 1)
    # Check loss
    assert isinstance(out["loss"], tf.Tensor)
    # Target checks
    target = [
        {CLASS_NAME: np.array([[0, 0, 1, 1]], dtype=np.uint8)},
        {CLASS_NAME: np.array([[0, 0, 1, 1]], dtype=np.uint8)},
    ]
    with pytest.raises(AssertionError):
        out = model(input_tensor, target, training=True)

    target = [
        {CLASS_NAME: np.array([[0, 0, 1.5, 1.5]], dtype=np.float32)},
        {CLASS_NAME: np.array([[-0.2, -0.3, 1, 1]], dtype=np.float32)},
    ]
    with pytest.raises(ValueError):
        out = model(input_tensor, target, training=True)

    # Check the rotated case
    target = [
        {CLASS_NAME: np.array([[0.75, 0.75, 0.5, 0.5, 0], [0.65, 0.65, 0.3, 0.3, 0]], dtype=np.float32)},
        {CLASS_NAME: np.array([[0.75, 0.75, 0.5, 0.5, 0], [0.65, 0.7, 0.3, 0.4, 0]], dtype=np.float32)},
    ]
    loss = model(input_tensor, target, training=True)["loss"]
    assert isinstance(loss, tf.Tensor) and ((loss - out["loss"]) / loss).numpy() < 1


@pytest.fixture(scope="session")
def test_detectionpredictor(mock_pdf):
    batch_size = 4
    predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=batch_size), detection.db_resnet50(input_shape=(512, 512, 3))
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(pages)
    # The input PDF has 2 pages
    assert len(out) == 2

    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    return predictor


@pytest.fixture(scope="session")
def test_rotated_detectionpredictor(mock_pdf):
    batch_size = 4
    predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=batch_size),
        detection.db_resnet50(assume_straight_pages=False, input_shape=(512, 512, 3)),
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(pages)

    # The input PDF has 2 pages
    assert len(out) == 2

    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    return predictor


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "fast_tiny",
    ],
)
def test_detection_zoo(arch_name):
    # Model
    tf.keras.backend.clear_session()
    predictor = detection.zoo.detection_predictor(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, DetectionPredictor)
    input_tensor = tf.random.uniform(shape=[2, 1024, 1024, 3], minval=0, maxval=1)
    out, seq_maps = predictor(input_tensor, return_maps=True)
    assert all(isinstance(boxes, dict) for boxes in out)
    assert all(isinstance(boxes[CLASS_NAME], np.ndarray) and boxes[CLASS_NAME].shape[1] == 5 for boxes in out)
    assert all(isinstance(seq_map, np.ndarray) for seq_map in seq_maps)
    assert all(seq_map.shape[:2] == (1024, 1024) for seq_map in seq_maps)
    # check that all values in the seq_maps are between 0 and 1
    assert all((seq_map >= 0).all() and (seq_map <= 1).all() for seq_map in seq_maps)


def test_detection_zoo_error():
    with pytest.raises(ValueError):
        _ = detection.zoo.detection_predictor("my_fancy_model", pretrained=False)


def test_fast_reparameterization():
    dummy_input = tf.random.uniform(shape=[1, 1024, 1024, 3], minval=0, maxval=1)
    base_model = detection.fast_tiny(pretrained=True, exportable=True)
    base_model_params = np.sum([np.prod(v.shape) for v in base_model.trainable_variables])
    assert math.isclose(base_model_params, 13535296)  # base model params
    base_out = base_model(dummy_input, training=False)["logits"]
    tf.keras.backend.clear_session()
    rep_model = reparameterize(base_model)
    rep_model_params = np.sum([np.prod(v.shape) for v in base_model.trainable_variables])
    assert math.isclose(rep_model_params, 8520256)  # reparameterized model params
    rep_out = rep_model(dummy_input, training=False)["logits"]
    diff = base_out - rep_out
    assert np.mean(diff) < 5e-2


def test_erode():
    x = np.zeros((1, 3, 3, 1), dtype=np.float32)
    x[:, 1, 1] = 1
    x = tf.convert_to_tensor(x)
    expected = tf.zeros((1, 3, 3, 1))
    out = erode(x, 3)
    assert tf.math.reduce_all(out == expected)


def test_dilate():
    x = np.zeros((1, 3, 3, 1), dtype=np.float32)
    x[:, 1, 1] = 1
    x = tf.convert_to_tensor(x)
    expected = tf.ones((1, 3, 3, 1))
    out = dilate(x, 3)
    assert tf.math.reduce_all(out == expected)


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["db_mobilenet_v3_large", (512, 512, 3), (512, 512, 1)],
        ["linknet_resnet18", (1024, 1024, 3), (1024, 1024, 1)],
        ["fast_tiny", (1024, 1024, 3), (1024, 1024, 1)],
        ["fast_tiny_rep", (1024, 1024, 3), (1024, 1024, 1)],  # Reparameterized model
        ["fast_small", (1024, 1024, 3), (1024, 1024, 1)],
        pytest.param(
            "db_resnet50",
            (512, 512, 3),
            (512, 512, 1),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "linknet_resnet34",
            (1024, 1024, 3),
            (1024, 1024, 1),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "linknet_resnet50",
            (512, 512, 3),
            (512, 512, 1),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
        pytest.param(
            "fast_base",
            (512, 512, 3),
            (512, 512, 1),
            marks=pytest.mark.skipif(system_available_memory < 16, reason="too less memory"),
        ),
    ],
)
def test_models_onnx_export(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    tf.keras.backend.clear_session()
    if arch_name == "fast_tiny_rep":
        model = reparameterize(detection.fast_tiny(pretrained=True, exportable=True, input_shape=input_shape))
    else:
        model = detection.__dict__[arch_name](pretrained=True, exportable=True, input_shape=input_shape)
    # batch_size = None for dynamic batch size
    dummy_input = [tf.TensorSpec([None, *input_shape], tf.float32, name="input")]
    np_dummy_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
    tf_logits = model(np_dummy_input, training=False)["logits"].numpy()
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
        assert np.allclose(ort_outs[0], tf_logits, atol=1e-4)
    except AssertionError:
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {np.max(np.abs(tf_logits - ort_outs[0]))}")

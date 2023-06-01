import os
import shutil
import tempfile

import numpy as np
import onnxruntime
import pytest
import tensorflow as tf

from doctr.io import DocumentFile
from doctr.models import recognition
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.crnn.tensorflow import CTCPostProcessor
from doctr.models.recognition.master.tensorflow import MASTERPostProcessor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.sar.tensorflow import SARPostProcessor
from doctr.models.recognition.vitstr.tensorflow import ViTSTRPostProcessor
from doctr.models.utils import export_model_to_onnx
from doctr.utils.geometry import extract_crops


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (32, 128, 3)],
        ["crnn_mobilenet_v3_small", (32, 128, 3)],
        ["crnn_mobilenet_v3_large", (32, 128, 3)],
        ["sar_resnet31", (32, 128, 3)],
        ["master", (32, 128, 3)],
        ["vitstr_small", (32, 128, 3)],
        ["vitstr_base", (32, 128, 3)],
    ],
)
def test_recognition_models(arch_name, input_shape):
    batch_size = 4
    reco_model = recognition.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    assert isinstance(reco_model, tf.keras.Model)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)
    target = ["i", "am", "a", "jedi"]

    out = reco_model(input_tensor, target, return_model_output=True, return_preds=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    assert isinstance(out["out_map"], tf.Tensor)
    assert out["out_map"].dtype == tf.float32
    assert isinstance(out["preds"], list)
    assert len(out["preds"]) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out["preds"])
    assert isinstance(out["loss"], tf.Tensor)
    # test model in train mode needs targets
    with pytest.raises(ValueError):
        reco_model(input_tensor, None, training=True)


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        [SARPostProcessor, [2, 30, 119]],
        [CTCPostProcessor, [2, 30, 119]],
        [MASTERPostProcessor, [2, 30, 119]],
        [ViTSTRPostProcessor, [2, 30, 119]],
    ],
)
def test_reco_postprocessors(post_processor, input_shape, mock_vocab):
    processor = post_processor(mock_vocab)
    decoded = processor(tf.random.uniform(shape=input_shape, minval=0, maxval=1, dtype=tf.float32))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == input_shape[0]
    assert all(char in mock_vocab for word, _ in decoded for char in word)
    # Repr
    assert repr(processor) == f"{post_processor.__name__}(vocab_size={len(mock_vocab)})"


@pytest.fixture(scope="session")
def test_recognitionpredictor(mock_pdf, mock_vocab):  # noqa: F811
    batch_size = 4
    predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=batch_size, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(vocab=mock_vocab, input_shape=(32, 128, 3)),
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    # Create bounding boxes
    boxes = np.array([[0.5, 0.5, 0.75, 0.75], [0.5, 0.5, 1.0, 1.0]], dtype=np.float32)
    crops = extract_crops(pages[0], boxes)

    out = predictor(crops)

    # One prediction per crop
    assert len(out) == boxes.shape[0]
    assert all(isinstance(val, str) and isinstance(conf, float) for val, conf in out)

    # Dimension check
    with pytest.raises(ValueError):
        input_crop = (255 * np.random.rand(1, 128, 64, 3)).astype(np.uint8)
        _ = predictor([input_crop])

    return predictor


@pytest.mark.parametrize(
    "arch_name",
    [
        "crnn_vgg16_bn",
        "crnn_mobilenet_v3_small",
        "crnn_mobilenet_v3_large",
        "sar_resnet31",
        "master",
        "vitstr_small",
        "vitstr_base",
    ],
)
def test_recognition_zoo(arch_name):
    batch_size = 2
    # Model
    predictor = recognition.zoo.recognition_predictor(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, RecognitionPredictor)
    input_tensor = tf.random.uniform(shape=[batch_size, 128, 128, 3], minval=0, maxval=1)
    out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) for word, conf in out)


@pytest.mark.parametrize(
    "arch_name",
    [
        "crnn_vgg16_bn",
        "crnn_mobilenet_v3_small",
        "crnn_mobilenet_v3_large",
    ],
)
def test_crnn_beam_search(arch_name):
    batch_size = 2
    # Model
    predictor = recognition.zoo.recognition_predictor(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, RecognitionPredictor)
    input_tensor = tf.random.uniform(shape=[batch_size, 128, 128, 3], minval=0, maxval=1)
    out = predictor(input_tensor, beam_width=10, top_paths=10)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(
        isinstance(words, list)
        and isinstance(confs, list)
        and all([isinstance(word, str) for word in words])
        and all([isinstance(conf, float) for conf in confs])
        for words, confs in out
    )


def test_recognition_zoo_error():
    with pytest.raises(ValueError):
        _ = recognition.zoo.recognition_predictor("my_fancy_model", pretrained=False)


@pytest.mark.skipif(os.getenv("SLOW", "0") == "0", reason="slow test")
@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (32, 128, 3)],
        ["crnn_mobilenet_v3_small", (32, 128, 3)],
        ["crnn_mobilenet_v3_large", (32, 128, 3)],
        ["sar_resnet31", (32, 128, 3)],
        ["master", (32, 128, 3)],
        ["vitstr_small", (32, 128, 3)],  # testing one vitstr version is enough
    ],
)
def test_models_onnx_export(arch_name, input_shape):
    # Model
    batch_size = 2
    tf.keras.backend.clear_session()
    model = recognition.__dict__[arch_name](pretrained=True, exportable=True, input_shape=input_shape)
    # SAR, MASTER, ViTSTR export currently only available with constant batch size
    if arch_name in ["sar_resnet31", "master", "vitstr_small"]:
        dummy_input = [tf.TensorSpec([batch_size, *input_shape], tf.float32, name="input")]
    else:
        # batch_size = None for dynamic batch size
        dummy_input = [tf.TensorSpec([None, *input_shape], tf.float32, name="input")]
    np_dummy_input = np.random.rand(batch_size, *input_shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path, output = export_model_to_onnx(
            model,
            model_name=os.path.join(tmpdir, "model"),
            dummy_input=dummy_input,
            large_model=True if arch_name == "master" else False,
        )
        assert os.path.exists(model_path)

        if arch_name == "master":
            # large models are exported as zip archive
            shutil.unpack_archive(model_path, tmpdir, "zip")
            model_path = os.path.join(tmpdir, "__MODEL_PROTO.onnx")
        else:
            model_path = os.path.join(tmpdir, "model.onnx")

        # Inference
        ort_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        ort_outs = ort_session.run(output, {"input": np_dummy_input})
        assert isinstance(ort_outs, list) and len(ort_outs) == 1
        assert ort_outs[0].shape[0] == batch_size

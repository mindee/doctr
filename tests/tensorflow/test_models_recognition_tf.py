import numpy as np
import pytest
import tensorflow as tf

from doctr.io import DocumentFile
from doctr.models import recognition
from doctr.models._utils import extract_crops
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.crnn.tensorflow import CTCPostProcessor
from doctr.models.recognition.master.tensorflow import MASTERPostProcessor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.sar.tensorflow import SARPostProcessor


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (32, 128, 3)],
        ["crnn_mobilenet_v3_small", (32, 128, 3)],
        ["crnn_mobilenet_v3_large", (32, 128, 3)],
        ["sar_resnet31", (32, 128, 3)],
        ["master", (32, 128, 3)],
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
    assert isinstance(out['out_map'], tf.Tensor)
    assert out['out_map'].dtype == tf.float32
    assert isinstance(out['preds'], list)
    assert len(out['preds']) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out['preds'])
    assert isinstance(out['loss'], tf.Tensor)


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        [SARPostProcessor, [2, 30, 119]],
        [CTCPostProcessor, [2, 30, 119]],
        [MASTERPostProcessor, [2, 30, 119]],
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
    assert repr(processor) == f'{post_processor.__name__}(vocab_size={len(mock_vocab)})'


@pytest.fixture(scope="session")
def test_recognitionpredictor(mock_pdf, mock_vocab):  # noqa: F811

    batch_size = 4
    predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=batch_size, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(vocab=mock_vocab, input_shape=(32, 128, 3))
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    # Create bounding boxes
    boxes = np.array([[.5, .5, 0.75, 0.75], [0.5, 0.5, 1., 1.]], dtype=np.float32)
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
        "master"
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


def test_recognition_zoo_error():
    with pytest.raises(ValueError):
        _ = recognition.zoo.recognition_predictor("my_fancy_model", pretrained=False)

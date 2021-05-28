import pytest
import numpy as np
import tensorflow as tf

from doctr.models import recognition, PreProcessor
from doctr.documents import DocumentFile
from doctr.models import extract_crops


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (32, 128, 3)],
        ["sar_vgg16_bn", (32, 128, 3)],
        ["sar_resnet31", (32, 128, 3)],
        ["crnn_resnet31", (32, 128, 3)],
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
    assert isinstance(out['preds'], list)
    assert len(out['preds']) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out['preds'])
    assert isinstance(out['out_map'], tf.Tensor)
    assert isinstance(out['loss'], tf.Tensor)


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        ["SARPostProcessor", [2, 30, 119]],
        ["CTCPostProcessor", [2, 30, 119]],
    ],
)
def test_reco_postprocessors(post_processor, input_shape, mock_vocab):
    processor = recognition.__dict__[post_processor](mock_vocab)
    decoded = processor(tf.random.uniform(shape=input_shape, minval=0, maxval=1, dtype=tf.float32))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == input_shape[0]
    assert all(char in mock_vocab for word, _ in decoded for char in word)
    # Repr
    assert repr(processor) == f'{post_processor}(vocab_size={len(mock_vocab)})'


@pytest.fixture(scope="session")
def test_recognitionpredictor(mock_pdf, mock_vocab):  # noqa: F811

    batch_size = 4
    predictor = recognition.RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=batch_size, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(vocab=mock_vocab, input_shape=(32, 128, 3))
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    # Create bounding boxes
    boxes = np.array([[0, 0, 0.25, 0.25], [0.5, 0.5, 1., 1.]], dtype=np.float32)
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
        "sar_vgg16_bn",
        "sar_resnet31",
        "crnn_resnet31",
    ],
)
def test_recognition_zoo(arch_name):
    batch_size = 2
    # Model
    predictor = recognition.zoo.recognition_predictor(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, recognition.RecognitionPredictor)
    input_tensor = tf.random.uniform(shape=[batch_size, 1024, 1024, 3], minval=0, maxval=1)
    out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) for word, conf in out)


def test_recognition_zoo_error():
    with pytest.raises(ValueError):
        _ = recognition.zoo.recognition_predictor("my_fancy_model", pretrained=False)

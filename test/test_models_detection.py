import pytest
import numpy as np
import tensorflow as tf

from doctr.models import detection, PreProcessor
from doctr.documents import DocumentFile


def test_dbpostprocessor():
    postprocessor = detection.DBPostProcessor()
    mock_batch = tf.random.uniform(shape=[2, 512, 512, 1], minval=0, maxval=1)
    out = postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:4] >= 0, sample[:4] <= 1)) for sample in out)
    # Repr
    assert repr(postprocessor) == 'DBPostProcessor(box_thresh=0.1, max_candidates=1000)'
    # Edge case when the expanded points of the polygon has two lists
    issue_points = np.array([
        [869, 561],
        [923, 581],
        [925, 595],
        [915, 583],
        [889, 583],
        [905, 593],
        [882, 601],
        [901, 595],
        [904, 604],
        [876, 608],
        [915, 614],
        [911, 605],
        [925, 601],
        [930, 616],
        [911, 617],
        [900, 636],
        [931, 637],
        [904, 649],
        [932, 649],
        [932, 628],
        [918, 627],
        [934, 624],
        [935, 573],
        [909, 569],
        [934, 562]], dtype=np.int32)
    out = postprocessor.polygon_to_box(issue_points)
    assert isinstance(out, tuple) and len(out) == 4


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet50", (1024, 1024, 3), (1024, 1024, 1), True],
        ["linknet", (1024, 1024, 3), (1024, 1024, 1), False],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob):
    batch_size = 2
    model = detection.__dict__[arch_name](pretrained=True)
    assert isinstance(model, tf.keras.Model)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)
    target = [
        dict(boxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 1, 1]], dtype=np.float32), flags=[True, False]),
        dict(boxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 1, 1]], dtype=np.float32), flags=[True, False])
    ]
    # test training model
    out = model(input_tensor, target, return_model_output=True, return_boxes=True, training=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    # Check proba map
    seg_map = out['out_map'].numpy()
    assert seg_map.shape == (batch_size, *output_size)
    if out_prob:
        assert np.all(np.logical_and(seg_map >= 0, seg_map <= 1))
    # Check boxes
    for boxes in out['boxes']:
        assert boxes.shape[1] == 5
        assert np.all(boxes[:, :2] < boxes[:, 2:4])
        assert np.all(boxes[:, :4] >= 0) and np.all(boxes[:, :4] <= 1)
    # Check loss
    assert isinstance(out['loss'], tf.Tensor)


@pytest.fixture(scope="session")
def test_detectionpredictor(mock_pdf):  # noqa: F811

    batch_size = 4
    predictor = detection.DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=batch_size),
        detection.db_resnet50(input_shape=(512, 512, 3))
    )

    pages = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(pages)

    # The input PDF has 8 pages
    assert len(out) == 8

    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    return predictor


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet50",
        "linknet",
    ],
)
def test_detection_zoo(arch_name):
    # Model
    predictor = detection.zoo.detection_predictor(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, detection.DetectionPredictor)
    input_tensor = tf.random.uniform(shape=[2, 1024, 1024, 3], minval=0, maxval=1)
    out = predictor(input_tensor)
    assert isinstance(out, list)
    assert all(isinstance(boxes, np.ndarray) and boxes.shape[1] == 5 for boxes in out)


def test_detection_zoo_error():
    with pytest.raises(ValueError):
        _ = detection.zoo.detection_predictor("my_fancy_model", pretrained=False)


def test_linknet_postprocessor():
    postprocessor = detection.LinkNetPostProcessor()
    mock_batch = tf.random.uniform(shape=[2, 512, 512, 1], minval=0, maxval=1)
    out = postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:4] >= 0, sample[:4] <= 1)) for sample in out)

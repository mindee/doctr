import pytest
import math
import numpy as np
import tensorflow as tf

from doctr.models import detection
from doctr.documents import DocumentFile


def test_detpreprocessor(mock_pdf):  # noqa: F811
    num_docs = 3
    batch_size = 4
    docs = [DocumentFile.from_pdf(mock_pdf).as_images() for _ in range(num_docs)]
    processor = detection.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])

    # Number of batches
    assert len(batched_docs) == math.ceil(8 * num_docs / batch_size)
    # Total number of samples
    assert sum(batch.shape[0] for batch in batched_docs) == 8 * num_docs
    # Batch size
    assert all(batch.shape[0] == batch_size for batch in batched_docs[:-1])
    assert batched_docs[-1].shape[0] == batch_size if (8 * num_docs) % batch_size == 0 else (8 * num_docs) % batch_size
    # Data type
    assert all(batch.dtype == tf.float32 for batch in batched_docs)
    # Image size
    assert all(batch.shape[1:] == (512, 512, 3) for batch in batched_docs)
    # Test with non-full last batch
    batch_size = 16
    processor = detection.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])
    assert batched_docs[-1].shape[0] == (8 * num_docs) % batch_size

    # Repr
    assert repr(processor) == 'DetectionPreProcessor(output_size=(512, 512), mean=[0.5 0.5 0.5], std=[1. 1. 1.])'


def test_dbpostprocessor():
    postprocessor = detection.DBPostProcessor()
    mock_batch = dict(proba_map=tf.random.uniform(shape=[2, 512, 512, 1], minval=0, maxval=1))
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


def test_db_resnet50_training_mode():
    model = detection.db_resnet50(pretrained=False)
    assert isinstance(model, tf.keras.Model)
    dbinput = tf.random.uniform(shape=[2, 1024, 1024, 3], minval=0, maxval=1)
    # test training model
    dboutput_train = model(dbinput, training=True)
    assert isinstance(dboutput_train, dict)
    assert len(dboutput_train) == 3
    assert all(
        np.all(np.logical_and(out_map.numpy() >= 0, out_map.numpy() <= 1)) for out_map in dboutput_train.values()
    )
    # batch size
    assert all(out.numpy().shape == (2, 1024, 1024, 1) for out in dboutput_train.values())


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
    # test prediction model
    out = model(input_tensor)
    assert isinstance(out, dict)
    out = out["proba_map"].numpy()
    assert out.shape == (batch_size, *output_size)
    if out_prob:
        assert np.all(np.logical_and(out >= 0, out <= 1))



@pytest.fixture(scope="session")
def test_detectionpredictor(mock_pdf):  # noqa: F811

    batch_size = 4
    predictor = detection.DetectionPredictor(
        detection.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size),
        detection.db_resnet50(input_shape=(512, 512, 3)),
        detection.DBPostProcessor()
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


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet50",
        "linknet"
    ],
)
def test_compute_detection_loss(arch_name):
    boxes = [
        np.array([[0.003, 0.002, 0.04, 0.03], [0.3, 0.1, 0.4, 0.3]]),
        np.array([[0.003, 0.002, 0.04, 0.03], [0.3, 0.1, 0.4, 0.3]]),
        np.zeros((0, 4)),
    ]
    flags = [[True, False], [True, False], []]
    model = detection.__dict__[arch_name](pretrained=True)
    model_input = tf.random.uniform(shape=[3, 1024, 1024, 3], minval=0, maxval=1)
    output = model(model_input, training=True)
    loss = model.compute_loss(
        model_output=output,
        batch_boxes=boxes,
        batch_flags=flags,
    )
    assert isinstance(loss, tf.Tensor)


def test_linknet_postprocessor():
    postprocessor = detection.LinkNetPostProcessor()
    mock_batch = dict(proba_map=tf.random.uniform(shape=[2, 512, 512, 1], minval=0, maxval=1))
    out = postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:4] >= 0, sample[:4] <= 1)) for sample in out)

def test_db_freezing():
    model = detection.db_resnet50(pretrained=True, freeze_bckb=False)
    freezed_model = detection.db_resnet50(pretrained=True, freeze_bckb=True)
    assert len(model.trainable_weights) > len(freezed_model.trainable_weights)

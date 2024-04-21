import numpy as np
import pytest

from doctr.contrib import artefacts
from doctr.contrib.base import _BasePredictor
from doctr.io import DocumentFile


def test_base_predictor():
    # check that we need to provide either a url or a model_path
    with pytest.raises(ValueError):
        _ = _BasePredictor(batch_size=2)

    predictor = _BasePredictor(batch_size=2, url=artefacts.default_cfgs["yolov8_artefact"]["url"])
    # check that we need to implement preprocess and postprocess
    with pytest.raises(NotImplementedError):
        predictor.preprocess(np.zeros((10, 10, 3)))
    with pytest.raises(NotImplementedError):
        predictor.postprocess([np.zeros((10, 10, 3))], [[np.zeros((10, 10, 3))]])


def test_artefact_detector(mock_artefact_image_stream):
    doc = DocumentFile.from_images([mock_artefact_image_stream])
    detector = artefacts.ArtefactDetector(batch_size=2, conf_threshold=0.5, iou_threshold=0.5)
    results = detector(doc)
    assert isinstance(results, list) and len(results) == 1 and isinstance(results[0], list)
    assert all(isinstance(artefact, dict) for artefact in results[0])
    # check result keys
    assert all(key in results[0][0] for key in ["label", "confidence", "box"])
    assert all(len(artefact["box"]) == 4 for artefact in results[0])
    assert all(isinstance(coord, int) for box in results[0] for coord in box["box"])
    assert all(isinstance(artefact["confidence"], float) for artefact in results[0])
    assert all(isinstance(artefact["label"], str) for artefact in results[0])
    # check results for the mock image are 9 artefacts
    assert len(results[0]) == 9
    # test visualization non-blocking for tests
    detector.show(block=False)

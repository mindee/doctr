from app.schemas import DetectionIn, KIEIn, OCRIn, RecognitionIn
from app.vision import init_predictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.kie_predictor import KIEPredictor
from doctr.models.predictor import OCRPredictor
from doctr.models.recognition.predictor import RecognitionPredictor


def test_vision():
    assert isinstance(init_predictor(OCRIn()), OCRPredictor)
    assert isinstance(init_predictor(DetectionIn()), DetectionPredictor)
    assert isinstance(init_predictor(RecognitionIn()), RecognitionPredictor)
    assert isinstance(init_predictor(KIEIn()), KIEPredictor)

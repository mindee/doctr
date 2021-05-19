import os
from doctr.models.artefacts import QRCodeDetector
import numpy as np


def test_qr_code_detector(mock_image_folder):
    detector = QRCodeDetector()
    image = np.random.random_sample((512, 512, 3))
    qrcode = detector(image)
    assert qrcode is None

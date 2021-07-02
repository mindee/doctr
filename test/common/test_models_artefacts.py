import os
import numpy as np
from doctr.models.artefacts import BarCodeDetector, FaceDetector, QRCodeDetector
import cv2


def test_qr_code_detector(mock_image_folder):
    detector = QRCodeDetector()
    for img in os.listdir(mock_image_folder):
        image = cv2.imread(os.path.join(mock_image_folder, img)).astype(np.uint8)
        qrcode = detector(image)
    assert qrcode is None


def test_bar_code_detector(mock_image_folder):
    detector = BarCodeDetector()
    for img in os.listdir(mock_image_folder):
        image = cv2.imread(os.path.join(mock_image_folder, img)).astype(np.uint8)
        barcode = detector(image)
    assert len(barcode) == 0


def test_face_detector(mock_image_folder):
    detector = FaceDetector(n_faces=1)
    image = np.random.randint(low=0, high=255, size=(512, 512, 3), dtype=np.uint8)
    faces = detector(image)
    assert len(faces) <= 1

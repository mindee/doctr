import os
from doctr.models.artefacts import QRCodeDetector, FaceDetector
from doctr.documents import DocumentFile


def test_qr_code_detector(mock_image_folder):
    detector = QRCodeDetector()
    for img in os.listdir(mock_image_folder):
        image = DocumentFile.from_images(os.path.join(mock_image_folder, img))[0]
        qrcode = detector(image)
        assert qrcode is None


def test_face_detector(mock_image_folder):
    detector = FaceDetector(n_faces=1)
    for img in os.listdir(mock_image_folder):
        image = DocumentFile.from_images(os.path.join(mock_image_folder, img))[0]
        faces = detector(image)
        assert len(faces) <= 1

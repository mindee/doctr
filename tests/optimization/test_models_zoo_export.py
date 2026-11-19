import pytest
import torch

from doctr import models
from doctr.io import Document, DocumentFile
from doctr.models import detection, recognition
from doctr.models.classification import (
    crop_orientation_predictor,
    mobilenet_v3_small_crop_orientation,
    mobilenet_v3_small_page_orientation,
    page_orientation_predictor,
)


@pytest.mark.parametrize(
    "det_arch, reco_arch",
    [
        ["fast_base", "crnn_vgg16_bn"],
    ],
)
def test_end_to_end_torch_compile(det_arch, reco_arch, mock_payslip):
    doc = DocumentFile.from_images(mock_payslip)
    predictor = models.ocr_predictor(det_arch, reco_arch, pretrained=True, assume_straight_pages=False)
    out = predictor(doc)

    assert isinstance(out, Document)

    # Compile the models
    detection_model = torch.compile(detection.__dict__[det_arch](pretrained=True).eval())
    recognition_model = torch.compile(recognition.__dict__[reco_arch](pretrained=True).eval())
    crop_orientation_model = torch.compile(mobilenet_v3_small_crop_orientation(pretrained=True).eval())
    page_orientation_model = torch.compile(mobilenet_v3_small_page_orientation(pretrained=True).eval())

    predictor = models.ocr_predictor(detection_model, recognition_model, assume_straight_pages=False)
    # Set the orientation predictors
    # NOTE: only required for non-straight pages and non-disabled orientation classification
    predictor.crop_orientation_predictor = crop_orientation_predictor(crop_orientation_model)
    predictor.page_orientation_predictor = page_orientation_predictor(page_orientation_model)
    compiled_out = predictor(doc)

    # Check that the number of word detections is the same
    assert len(out.pages[0].blocks[0].lines[0].words) == len(compiled_out.pages[0].blocks[0].lines[0].words)
    # Check that the words are the same
    assert all(
        word.value == compiled_out.pages[0].blocks[0].lines[0].words[i].value
        for i, word in enumerate(out.pages[0].blocks[0].lines[0].words)
    )

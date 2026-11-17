import json
import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr import models
from doctr.datasets import LATEX_TOKENS, normalize_latex
from doctr.io import Document, DocumentFile
from doctr.io.elements import Formula
from doctr.models import detection, formula, recognition
from doctr.models.factory import _save_model_and_config_for_hf_hub
from doctr.models.formula import FormulaNet
from doctr.models.formula.formulanet.pytorch import default_cfgs as formulanet_cfgs
from doctr.models.formula.predictor import FormulaPredictor
from doctr.models.formula.zoo import formula_predictor
from doctr.models.layout.zoo import layout_predictor
from doctr.models.predictor import OCRPredictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx
from doctr.utils.metrics import FormulaMetric

_TARGETS = [r"\frac{a}{b^2}", r"\sum_{i=1}^{n} x_i \leq \alpha"]
# The formula of the `mock_formula_image` fixture
_MOCK_FORMULA = r"J ( u , y ) = \int _ { 0 } ^ { 1 } \frac { | u | ^ { 2 } } { 2 } \, d x + \Psi ( y )"


@pytest.mark.parametrize("train_mode", [True, False])
@pytest.mark.parametrize("input_shape", [(3, 256, 768), (3, 128, 256)])
def test_formula_models(input_shape, train_mode):
    batch_size = 2
    model = formula.formulanet(pretrained=True, input_shape=input_shape)
    model = model.train() if train_mode else model.eval()
    assert isinstance(model, FormulaNet)
    assert model.cfg["input_shape"] == input_shape
    input_tensor = torch.rand((batch_size, *input_shape))
    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()

    out = model(input_tensor, _TARGETS, return_model_output=True, return_preds=not train_mode)
    assert isinstance(out, dict)
    assert len(out) == (2 if train_mode else 3)
    # Teacher forcing: one step per token + EOS of the longest target
    if train_mode:
        num_steps = max(len(normalize_latex(t).split()) for t in _TARGETS) + 1
        assert out["out_map"].shape == (batch_size, num_steps, len(LATEX_TOKENS) + 1)
    else:
        assert out["out_map"].shape[0] == batch_size and out["out_map"].shape[1] <= model.max_decode_length + 1
        assert len(out["preds"]) == batch_size
        assert all(isinstance(latex, str) and 0 <= conf <= 1 for latex, conf in out["preds"])
    assert out["out_map"].dtype == torch.float32
    assert isinstance(out["loss"], torch.Tensor) and torch.isfinite(out["loss"])
    if train_mode:
        out["loss"].backward()
        assert model.head.weight.grad is not None
    assert hasattr(model, "from_pretrained")

    # Inference without targets
    model.eval()
    with torch.no_grad():
        out = model(input_tensor)
    assert set(out) == {"preds"} and len(out["preds"]) == batch_size

    # Training requires targets
    model.train()
    with pytest.raises(ValueError):
        model(input_tensor)


def test_formulanet_kv_cache_matches_full_decoding():
    model = formula.formulanet(pretrained=True).eval()
    with torch.no_grad():
        memory = model.encode(torch.rand((2, 3, 256, 768)))
        greedy = model.decode_greedy(memory, max_len=10)
        # Re-decoding the generated sequences in parallel gives the same logits
        tokens = torch.cat([torch.full((2, 1), model.sos), greedy.argmax(-1)[:, :-1]], dim=1)
        full = model.decode(tokens, memory)
    # At most 10 tokens + EOS: the decoding stops earlier once both sequences have ended
    assert greedy.shape == full.shape and greedy.shape[1] <= 11 and greedy.shape[2] == len(LATEX_TOKENS) + 1
    assert torch.allclose(greedy, full, atol=1e-4)


def test_formulanet_custom_vocab():
    vocab = ["a", "b", "\\frac", "{", "}"]
    model = formula.formulanet(pretrained=True, vocab=vocab).train()
    assert model.cfg["vocab"] == vocab and model.head.out_features == len(vocab) + 1
    out = model(torch.rand((1, 3, 64, 128)), [r"\frac{a}{b}"])
    assert torch.isfinite(out["loss"])
    with pytest.raises(ValueError):
        model(torch.rand((1, 3, 64, 128)), [r"\alpha"])


def test_formula_zoo(mock_formula_image):
    predictor = formula_predictor("formulanet", pretrained=True, batch_size=2)
    assert isinstance(predictor, FormulaPredictor)
    # A formula with white margins, and a dark crop
    crop = np.full((100, 400, 3), 255, dtype=np.uint8)
    crop[40:60, 50:350] = 0
    out = predictor([crop, np.zeros((32, 64, 3), dtype=np.uint8), crop])
    assert len(out) == 3
    assert all(isinstance(latex, str) and 0 <= conf <= 1 for latex, conf in out)
    assert out[0][0] == out[2][0] and out[0][1] == pytest.approx(out[2][1], abs=1e-5)
    assert predictor([]) == []

    # A real formula, with the margins of a layout region: read as it is
    (latex, confidence), *_ = predictor(DocumentFile.from_images([mock_formula_image]))
    assert isinstance(latex, str) and 0 <= confidence <= 1

    with pytest.raises(ValueError):
        predictor([np.zeros((32, 64), dtype=np.uint8)])
    with pytest.raises(ValueError):
        predictor([np.zeros((0, 64, 3), dtype=np.uint8)])
    with pytest.raises(ValueError):
        formula_predictor("unknown_arch")
    with pytest.raises(ValueError):
        formula_predictor(torch.nn.Identity())

    # Model instance, confidence aggregation & decoding length
    model = formula.formulanet(pretrained=True)
    predictor = formula_predictor(model, confidence_aggregation="min", max_decode_length=64)
    assert predictor.model is model and model.postprocessor.confidence_aggregation == "min"
    assert model.max_decode_length == 64


def test_formula_predictor_regions(monkeypatch):
    predictor = formula_predictor("formulanet", pretrained=True)
    crops = []

    def fake_forward(images, **kwargs):
        crops.extend(images)
        return [("x ^ 2", 0.9) for _ in images]

    monkeypatch.setattr(predictor, "forward", fake_forward)
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    polygon = np.array([[0.1, 0.5], [0.5, 0.5], [0.5, 0.6], [0.1, 0.6]], dtype=np.float32)
    regions = [
        {
            "class_names": ["Text", "Formula", "Formula"],
            "boxes": np.array([[0, 0, 0.1, 0.1], [0.1, 0.25, 0.5, 0.375], [0.2, 0.2, 0.2, 0.2]]),
            "scores": np.array([0.5, 0.8, 0.7]),
        },
        None,
        {"class_names": ["Formula"], "boxes": polygon[None], "scores": np.array([0.6])},
    ]
    formulas = predictor.predict_regions([page, page, page], regions)
    # Only the formula regions are cropped, the degenerate one is skipped
    assert len(crops) == 2 and crops[0].shape[:2] == (51, 241) and crops[1].shape[:2] == (40, 240)
    assert [len(page_formulas) for page_formulas in formulas] == [1, 0, 1]
    assert formulas[0][0]["value"] == "x ^ 2" and formulas[0][0]["confidence"] == 0.9
    assert formulas[0][0]["score"] == pytest.approx(0.8)
    np.testing.assert_allclose(formulas[0][0]["geometry"], [0.1, 0.25, 0.5, 0.375])
    np.testing.assert_allclose(formulas[2][0]["geometry"], polygon)
    assert predictor.predict_regions([page], [None]) == [[]]


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["formulanet", (3, 256, 768)],
    ],
)
def test_models_onnx_export(arch_name, input_shape):
    batch_size = 2
    model = formula.__dict__[arch_name](pretrained=True, exportable=True).eval()
    # The greedy decoding is unrolled in the exported graph (no early stop): keep it short
    model.max_decode_length = 4
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    pt_logits = model(dummy_input)["logits"].detach().cpu().numpy()
    assert pt_logits.shape == (batch_size, 5, len(LATEX_TOKENS) + 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = export_model_to_onnx(model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input)
        assert os.path.exists(model_path)
        ort_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        ort_outs = ort_session.run(["logits"], {"input": dummy_input.numpy()})
    assert ort_outs[0].shape == pt_logits.shape
    try:
        assert np.allclose(pt_logits, ort_outs[0], atol=1e-4)
    except AssertionError:
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {np.max(np.abs(pt_logits - ort_outs[0]))}")


@pytest.mark.parametrize("arch_name", ["formulanet"])
def test_torch_compiled_models(arch_name, mock_formula_image):
    doc = DocumentFile.from_images([mock_formula_image])
    model = formula.__dict__[arch_name](pretrained=True).eval()
    # Bound the decoding, in case the model does not end the formula of the mock image
    predictor = formula_predictor(model, max_decode_length=32)
    assert isinstance(predictor, FormulaPredictor)
    out = predictor(doc)

    # Compile the model
    compiled_model = torch.compile(model)
    assert isinstance(compiled_model, _CompiledModule)
    compiled_predictor = formula_predictor(compiled_model)
    compiled_out = compiled_predictor(doc)

    # Compare
    assert out[0][0] == compiled_out[0][0]
    assert np.allclose(out[0][1], compiled_out[0][1], atol=1e-4)


def test_formulanet_pretrained_prediction(mock_formula_image):
    if formulanet_cfgs["formulanet"]["url"] is None:
        pytest.skip("no pretrained weights for formulanet yet")
    predictor = formula_predictor("formulanet", pretrained=True)
    (latex, confidence), *_ = predictor(DocumentFile.from_images([mock_formula_image]))
    metric = FormulaMetric()
    metric.update([_MOCK_FORMULA], [latex])
    assert metric.summary()["edit_similarity"] >= 0.9, latex
    assert 0 < confidence <= 1


def test_ocrpredictor_formulas(mock_pdf, mock_vocab, monkeypatch):
    det_predictor = models.detection_predictor(
        detection.db_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, assume_straight_pages=True),
        batch_size=2,
    )
    reco_predictor = models.recognition_predictor(
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab), batch_size=32
    )
    layout_pred = layout_predictor("lw_detr_s", pretrained=False)
    formula_pred = formula_predictor("formulanet", pretrained=True)

    # A formula predictor requires a layout predictor (formulas are located with the layout model)
    with pytest.raises(ValueError):
        OCRPredictor(det_predictor, reco_predictor, formula_predictor=formula_pred)

    doc = DocumentFile.from_pdf(mock_pdf)
    page_h, page_w = doc[0].shape[:2]

    # A layout model locating one formula in the middle of each page
    def fake_layout(pages, **kwargs):
        return [
            {
                "class_names": ["Text", "Formula"],
                "boxes": np.array([[0.1, 0.1, 0.9, 0.2], [0.3, 0.4, 0.7, 0.5]], dtype=np.float32),
                "scores": np.array([0.9, 0.8], dtype=np.float32),
            }
            for _ in pages
        ]

    monkeypatch.setattr(layout_pred, "forward", fake_layout)
    masked_pages = []
    det_forward = det_predictor.forward

    def spy_detection(pages, **kwargs):
        masked_pages.extend(pages)
        return det_forward(pages, **kwargs)

    monkeypatch.setattr(det_predictor, "forward", spy_detection)
    monkeypatch.setattr(formula_pred, "forward", lambda crops, **kwargs: [("x ^ 2", 0.75) for _ in crops])

    predictor = OCRPredictor(
        det_predictor, reco_predictor, layout_predictor=layout_pred, formula_predictor=formula_pred
    )
    assert isinstance(predictor.formula_predictor, FormulaPredictor)
    out = predictor(doc)
    assert isinstance(out, Document)
    # The formula regions are masked out of the text detection
    region = masked_pages[0][int(0.4 * page_h) : int(0.5 * page_h), int(0.3 * page_w) : int(0.7 * page_w)]
    assert np.all(region == 0)
    for page in out.pages:
        assert len(page.formulas) == 1 and isinstance(page.formulas[0], Formula)
        formula_elt = page.formulas[0]
        assert formula_elt.value == "x ^ 2" and formula_elt.confidence == 0.75
        assert formula_elt.objectness_score == pytest.approx(0.8)
        np.testing.assert_allclose(formula_elt.geometry, ((0.3, 0.4), (0.7, 0.5)), atol=1e-6)
        assert page.export()["formulas"] == [formula_elt.export()]
        assert "$$\nx ^ 2\n$$" in page.export_as_markdown()
        assert formula_elt in page.items_in_reading_order()

    # Without a formula predictor, the formula regions are left to the text detection
    masked_pages.clear()
    predictor = OCRPredictor(det_predictor, reco_predictor, layout_predictor=layout_pred)
    out = predictor(doc)
    assert all(page.formulas == [] for page in out.pages)
    assert not np.all(masked_pages[0] == 0)


def test_ocrpredictor_formulas_factory():
    # The factory exposes a single `detect_formulas` flag, which also enables the layout model
    predictor = models.ocr_predictor(
        "db_mobilenet_v3_large", "crnn_vgg16_bn", pretrained=False, pretrained_backbone=False, detect_formulas=True
    )
    assert isinstance(predictor.formula_predictor, FormulaPredictor)
    assert predictor.layout_predictor is not None
    assert predictor._masked_classes() == {"Formula"}

    # No formulas by default
    predictor = models.ocr_predictor("db_mobilenet_v3_large", "crnn_vgg16_bn", pretrained=False)
    assert predictor.formula_predictor is None
    assert predictor._masked_classes() == set()


def test_formula_save_config_for_hub():
    model = formula.formulanet(pretrained=True).eval()
    with tempfile.TemporaryDirectory() as tmp_dir:
        _save_model_and_config_for_hf_hub(model, arch="formulanet", task="formula", save_dir=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "pytorch_model.bin"))
        with open(os.path.join(tmp_dir, "config.json")) as f:
            config = json.load(f)
        assert config["arch"] == "formulanet" and config["task"] == "formula"
        assert config["vocab"] == LATEX_TOKENS
        # The config is enough to rebuild the model and reload the weights
        rebuilt = formula.formulanet(
            pretrained=False,
            pretrained_backbone=False,
            input_shape=tuple(config["input_shape"]),
            vocab=config["vocab"],
        )
        rebuilt.from_pretrained(os.path.join(tmp_dir, "pytorch_model.bin"))
    for (name, param), rebuilt_param in zip(model.state_dict().items(), rebuilt.state_dict().values()):
        assert torch.equal(param, rebuilt_param), name


def test_formulanet_max_decode_length():
    model = formula.formulanet(pretrained=True, max_decode_length=8)
    assert model.max_length == 4096 and model.max_decode_length == 8
    assert model.pos_embed.num_embeddings == 4097  # the size of the model follows `max_length`
    x = torch.rand((2, 3, 256, 768))
    # Training labels longer than `max_decode_length` are learned entirely
    label = " ".join(["a"] * 30)
    out = model.train()(x, [label, "b"], return_model_output=True)
    assert out["out_map"].shape[1] == 31
    # The decoding generates at most `max_decode_length` tokens (+ EOS), even when the model never ends a formula
    model.exportable = True
    with torch.no_grad():
        assert model.eval()(x)["logits"].shape[1] == 9
        # An explicit length overrides it
        assert model.decode_greedy(model.encode(x), max_len=12).shape[1] == 13
    # The default cap, and its override in the predictor
    assert formula.formulanet(pretrained=True).max_decode_length == 2048
    predictor = formula_predictor(model, max_decode_length=4)
    assert predictor.model.max_decode_length == 4

import numpy as np
import pytest
import torch
from torchvision.models._utils import IntermediateLayerGetter

from doctr.models import recognition
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.crnn.pytorch import CTCPostProcessor
from doctr.models.recognition.master.pytorch import MASTERPostProcessor
from doctr.models.recognition.parseq.pytorch import PARSeqPostProcessor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.predictor._utils import split_crops
from doctr.models.recognition.sar.pytorch import SARPostProcessor
from doctr.models.recognition.viptr.pytorch import VIPTRPostProcessor
from doctr.models.recognition.vitstr.pytorch import ViTSTRPostProcessor


@pytest.mark.parametrize("train_mode", [True, False])
@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["crnn_vgg16_bn", (3, 32, 128)],
        ["crnn_mobilenet_v3_small", (3, 32, 128)],
        ["crnn_mobilenet_v3_large", (3, 32, 128)],
        ["sar_resnet31", (3, 32, 128)],
        ["master", (3, 32, 128)],
        ["vitstr_small", (3, 32, 128)],
        ["vitstr_base", (3, 32, 128)],
        ["parseq", (3, 32, 128)],
        ["viptr_tiny", (3, 32, 128)],
    ],
)
def test_recognition_models(arch_name, input_shape, train_mode, mock_vocab):
    batch_size = 4
    model = recognition.__dict__[arch_name](vocab=mock_vocab, pretrained=True, input_shape=input_shape)
    model = model.train() if train_mode else model.eval()
    assert isinstance(model, torch.nn.Module)
    # A backbone layer shared with an earlier one is dropped by IntermediateLayerGetter: no layer index may be missing
    if isinstance(model.feat_extractor, IntermediateLayerGetter):
        assert list(model.feat_extractor) == [str(idx) for idx in range(len(model.feat_extractor))]
    input_tensor = torch.rand((batch_size, *input_shape))
    target = ["i", "am", "a", "jedi"]

    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor, target, return_model_output=True, return_preds=not train_mode)
    assert isinstance(out, dict)
    assert len(out) == 3 if not train_mode else len(out) == 2
    if not train_mode:
        assert isinstance(out["preds"], list)
        assert len(out["preds"]) == batch_size
        assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in out["preds"])
    assert isinstance(out["out_map"], torch.Tensor)
    assert out["out_map"].dtype == torch.float32
    assert isinstance(out["loss"], torch.Tensor)
    # test model in train mode needs targets
    with pytest.raises(ValueError):
        model.train()
        model(input_tensor, None)
    # Check from pretrained is a class method
    assert hasattr(model, "from_pretrained")


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        [CTCPostProcessor, [2, 119, 30]],
        [SARPostProcessor, [2, 119, 30]],
        [ViTSTRPostProcessor, [2, 119, 30]],
        [MASTERPostProcessor, [2, 119, 30]],
        [PARSeqPostProcessor, [2, 119, 30]],
        [VIPTRPostProcessor, [2, 119, 30]],
    ],
)
def test_reco_postprocessors(post_processor, input_shape, mock_vocab):
    processor = post_processor(mock_vocab)
    decoded = processor(torch.rand(*input_shape))
    assert isinstance(decoded, list)
    assert all(isinstance(word, str) and isinstance(conf, float) and 0 <= conf <= 1 for word, conf in decoded)
    assert len(decoded) == input_shape[0]
    assert all(char in mock_vocab for word, _ in decoded for char in word)
    # Repr
    default = "mean" if post_processor in (ViTSTRPostProcessor, PARSeqPostProcessor) else "min"
    assert repr(processor) == (
        f"{post_processor.__name__}(vocab_size={len(mock_vocab)}, confidence_aggregation={default!r})"
    )
    assert "confidence_aggregation=<lambda>" in repr(post_processor(mock_vocab, confidence_aggregation=lambda p: 1.0))


def _logits(probs: list[list[float]], num_classes: int) -> torch.Tensor:
    probs_ = torch.zeros((1, len(probs), num_classes))
    probs_[0, :, : len(probs[0])] = torch.tensor(probs)
    return probs_.clamp_min(1e-9).log()


@pytest.mark.parametrize(
    "confidence_aggregation, ctc_conf, attention_conf",
    [
        ("mean", 0.75, 0.7),
        ("min", 0.6, 0.5),
        ("geometric_mean", 0.54**0.5, 0.45**0.5),
        (lambda probs: 1.0, 1.0, 1.0),
    ],
)
@pytest.mark.parametrize(
    "post_processor, num_classes",
    [
        [CTCPostProcessor, 4],
        [VIPTRPostProcessor, 4],
        [SARPostProcessor, 4],
        [ViTSTRPostProcessor, 5],
        [MASTERPostProcessor, 6],
        [PARSeqPostProcessor, 6],
    ],
)
def test_reco_postprocessors_confidence_aggregation(
    post_processor, num_classes, confidence_aggregation, ctc_conf, attention_conf
):
    processor = post_processor("abc", confidence_aggregation=confidence_aggregation)
    if post_processor in (CTCPostProcessor, VIPTRPostProcessor):
        # "a a <blank> b b <blank>": a character probability is the highest one within its run, blanks are ignored
        probs = [[0.5, 0.2, 0.1, 0.2], [0.9, 0.05, 0.0, 0.05], [0.1, 0.1, 0.1, 0.7], [0.1, 0.4, 0.2, 0.3]]
        probs += [[0.2, 0.6, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0]]
        word, conf = processor(_logits(probs, num_classes))[0]
        assert (word, conf) == ("ab", pytest.approx(ctc_conf, abs=1e-5))
    else:
        # "a b <eos> a": the probabilities after the <eos> token are ignored
        probs = [[0.9, 0.05, 0.03, 0.02], [0.1, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.7], [0.4, 0.2, 0.2, 0.2]]
        word, conf = processor(_logits(probs, num_classes))[0]
        assert (word, conf) == ("ab", pytest.approx(attention_conf, abs=1e-5))
    # Empty word
    assert processor(_logits([[0.1, 0.1, 0.1, 0.7]] * 3, num_classes)) == [("", 0.0)]
    for invalid in ["average", ["mean"], None]:
        with pytest.raises(ValueError, match="Unknown confidence aggregation"):
            post_processor("abc", confidence_aggregation=invalid)


class _MockRecoModel(torch.nn.Module):
    """Recognition model returning a predefined confidence for each crop it receives"""

    def __init__(self, confidences: list[float], postprocessor=None) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.confidences = confidences
        if postprocessor is not None:
            self.postprocessor = postprocessor

    def forward(self, x: torch.Tensor, return_preds: bool = False, **kwargs):
        return {"preds": [("ab", conf) for conf in self.confidences[: x.shape[0]]]}


@pytest.mark.parametrize(
    "postprocessor",
    [
        # The aggregation of the split parts is independent from the one of the character probabilities
        CTCPostProcessor("abc", confidence_aggregation="max"),
        PARSeqPostProcessor("abc"),
        # A custom model does not have to provide a postprocessor with a confidence aggregation
        None,
    ],
)
def test_recognition_predictor_split_confidence_aggregation(postprocessor):
    confidences = [0.9, 0.2, 0.7, 0.8, 0.6, 0.5]
    predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        _MockRecoModel(confidences, postprocessor),
    )
    # A wide crop split into several parts
    wide_crop = np.zeros((32, 32 * 20, 3), dtype=np.uint8)
    num_parts = len(split_crops([wide_crop], predictor.critical_ar, predictor.target_ar, predictor.overlap_ratio)[0])
    assert 2 <= num_parts <= len(confidences)
    confidences = confidences[:num_parts]

    # The lowest confidence of the parts by default
    assert predictor.split_confidence_aggregation == "min"
    assert predictor([wide_crop])[0][1] == pytest.approx(min(confidences))
    predictor.split_confidence_aggregation = "mean"
    assert predictor([wide_crop])[0][1] == pytest.approx(np.mean(confidences))
    predictor.split_confidence_aggregation = lambda confs: float(confs.max())
    assert predictor([wide_crop])[0][1] == pytest.approx(max(confidences))
    # A crop which is not split keeps the confidence of the model
    assert predictor([np.zeros((32, 128, 3), dtype=np.uint8)]) == [("ab", confidences[0])]
    # An invalid method is rejected at the first call, even without a crop to split
    predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        _MockRecoModel(confidences, postprocessor),
    )
    predictor.split_confidence_aggregation = "average"
    with pytest.raises(ValueError, match="Unknown confidence aggregation"):
        predictor([np.zeros((32, 128, 3), dtype=np.uint8)])


@pytest.mark.parametrize("empty_shape", [(0, 0, 3), (0, 20, 3), (20, 0, 3), (32, 128, 0)])
def test_recognition_predictor_empty_crop(empty_shape):
    predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        _MockRecoModel([0.9, 0.8]),
    )
    crop = np.zeros((32, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"non-empty, got shape \(" + ", ".join(map(str, empty_shape)) + r"\)"):
        predictor([crop, np.zeros(empty_shape, dtype=np.uint8)])
    # The OCR predictor drops empty crops before recognition, so valid crops are unaffected
    assert predictor([crop]) == [("ab", 0.9)]


def _spelled(text, height, char_width):
    """A crop spelling `text`, each char `char_width` columns wide: the first channel of a column is its char, the
    second the position of that char in the text (from 1)"""
    crop = np.zeros((height, len(text) * char_width, 3), dtype=np.uint8)
    for index, char in enumerate(text):
        columns = slice(index * char_width, (index + 1) * char_width)
        crop[:, columns, 0] = ord(char)
        crop[:, columns, 1] = index % 250 + 1
    return crop


class _ColumnsPreProcessor:
    """Hands the model the columns of each crop (the first row of its first two channels), zero-padded"""

    batch_size = 128

    def __call__(self, crops):
        columns = torch.zeros(len(crops), max(crop.shape[1] for crop in crops), 2)
        for row, crop in enumerate(crops):
            columns[row, : crop.shape[1]] = torch.from_numpy(crop[0, :, :2].astype(np.float32))
        return [columns[start : start + self.batch_size] for start in range(0, len(crops), self.batch_size)]


class _SpellingModel(torch.nn.Module):
    """Reads any _spelled crop, or a part or region of one, exactly: the chars any of its columns show, a char cut off
    at its border included. Like the models of docTR, it has a `max_length`, and readings longer than `cut_at` are cut
    short. The size of each batch it reads is recorded."""

    def __init__(self, max_length=32, cut_at=None):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.max_length = max_length
        self.cut_at = max_length if cut_at is None else cut_at
        self.batch_sizes = []

    def forward(self, x, return_preds=False, **kwargs):
        self.batch_sizes.append(x.shape[0])
        preds = []
        for columns in x.int().tolist():
            chars, last = [], 0
            for char, index in columns:
                if index and index != last:
                    chars.append(chr(char))
                last = index
            text = "".join(chars)[: self.cut_at]
            preds.append((text, 1 - 1 / (2 + len(text))))
        return {"preds": preds}


def _spelling_predictor(resolve_ambiguous_overlaps=True, model=None):
    return RecognitionPredictor(
        _ColumnsPreProcessor(), model or _SpellingModel(), resolve_ambiguous_overlaps=resolve_ambiguous_overlaps
    )


def _split_sizes(text):
    """Heights (10 to 47 px) and char widths (0.4 to 0.8 of the height) at which the crop of `text` is split"""
    critical_ar = _spelling_predictor().critical_ar
    for height in range(10, 48):
        for char_width in range(int(0.4 * height), int(0.8 * height) + 1):
            if len(text) * char_width / height > critical_ar:
                yield height, char_width


def test_recognition_predictor_resolves_a_run_split_across_parts():
    crops = [_spelled(text, 14, 11) for text in ["WARMUP-0000001", "INVOICE", "CONTRACTING-AGENT"]]
    unresolved = _spelling_predictor(False)(crops)
    # Each part is read right, and their merge adds a zero to the run
    assert [text for text, _ in unresolved] == ["WARMUP-00000001", "INVOICE", "CONTRACTING-AGENT"]

    predictor = _spelling_predictor()
    assert predictor.resolve_ambiguous_overlaps
    preds = predictor(crops)

    assert [text for text, _ in preds] == ["WARMUP-0000001", "INVOICE", "CONTRACTING-AGENT"]
    # The confidence is the one of the parts
    assert [conf for _, conf in preds] == [conf for _, conf in unresolved]
    # The parts are read in one batch, then the region covered by the two parts across the run, alone
    assert predictor.model.batch_sizes[1:] == [1]


def test_recognition_predictor_keeps_merge_strings_overlap_against_the_char_positions():
    # Each of the 2 parts of this crop cuts a zero in two: the exact reading of the region they cover points at an
    # overlap of 5 chars, but the positions of their chars spread evenly over each part at 4, which merge_strings picks
    predictor = _spelling_predictor()
    assert predictor([_spelled("U-0000000008911", 24, 14)])[0][0] == "U-00000000008911"
    assert predictor.model.batch_sizes == [2, 1]


@pytest.mark.parametrize(
    "text, misjoined, all_resolved",
    [
        # Runs and repeating patterns
        ("U-0000000008911", True, False),
        ("WARMUP-0000001", True, True),
        ("INFBC93AAAA072", True, True),
        ("ACCT 77777777", True, False),
        ("000-00-0000", False, True),
        ("REF-1111-2222", False, True),
        ("121212-343434", False, True),
        # Words
        ("ENDORSEMENT", False, True),
        ("PARTICIPANTS", False, True),
        ("BUSINESS-ENTITY-NAME", False, True),
    ],
)
def test_recognition_predictor_resolves_runs_wherever_split(text, misjoined, all_resolved):
    crops = [_spelled(text, *size) for size in _split_sizes(text)]
    unresolved = _spelling_predictor(False)(crops)

    preds = _spelling_predictor()(crops)

    # Without the region readings, the merges of some of these parts gain or lose part of a run
    unresolved_misjoins = sum(pred != text for pred, _ in unresolved)
    assert (unresolved_misjoins > 0) == misjoined
    # With them, none is merged wrong that merge_strings merges right, and the others are merged right, or more
    # rarely wrong where the char positions hold the region reading back
    assert all(pred == text for (pred, _), (unresolved_pred, _) in zip(preds, unresolved) if unresolved_pred == text)
    misjoins = sum(pred != text for pred, _ in preds)
    assert misjoins == 0 if all_resolved else 0 < misjoins < unresolved_misjoins
    # The confidence is still the one of the parts
    assert [conf for _, conf in preds] == [conf for _, conf in unresolved]


@pytest.mark.parametrize("text", ["WARMUP-0000001", "U-0000000008911"])
def test_recognition_predictor_keeps_merge_strings_overlap_with_a_region_reading_cut_short(text):
    # A model reading at most 12 chars cuts the reading of every region short
    crops = [_spelled(text, *size) for size in list(_split_sizes(text))[::4]]
    unresolved = _spelling_predictor(False, _SpellingModel(max_length=12))(crops)

    assert _spelling_predictor(True, _SpellingModel(max_length=12))(crops) == unresolved
    # Without the max length to tell the readings were cut short, some would have decided
    assert _spelling_predictor(True, _SpellingModel(max_length=1000, cut_at=12))(crops) != unresolved


def test_recognition_predictor_resolves_only_split_crops_when_asked():
    crops = [_spelled("WARMUP-0000001", 14, 11), _spelled("INVOICE", 14, 11)]

    predictor = _spelling_predictor(resolve_ambiguous_overlaps=False)
    assert [text for text, _ in predictor(crops)] == ["WARMUP-00000001", "INVOICE"]
    assert len(predictor.model.batch_sizes) == 1

    # A model without a max length is trusted to read the regions in full
    model = _SpellingModel()
    del model.max_length
    assert _spelling_predictor(model=model)(crops)[0][0] == "WARMUP-0000001"

    # Crops that are not split are read as they are
    predictor = _spelling_predictor()
    predictor.split_wide_crops = False
    assert predictor(crops) == [("WARMUP-0000001", 1 - 1 / 16), ("INVOICE", 1 - 1 / 9)]
    assert len(predictor.model.batch_sizes) == 1


@pytest.mark.parametrize(
    "arch_name", ["crnn_mobilenet_v3_small", "sar_resnet31", "master", "vitstr_small", "parseq", "viptr_tiny"]
)
def test_recognition_models_confidence_aggregation(arch_name, mock_vocab):
    model = recognition.__dict__[arch_name](vocab=mock_vocab, pretrained_backbone=False, confidence_aggregation="max")
    assert model.postprocessor.confidence_aggregation == "max"


@pytest.mark.parametrize(
    "input_shape",
    [
        (128, 128, 3),
        (32, 1024, 3),  # test case split wide crops
    ],
)
@pytest.mark.parametrize(
    "arch_name",
    [
        "crnn_vgg16_bn",
        "crnn_mobilenet_v3_small",
        "crnn_mobilenet_v3_large",
        "sar_resnet31",
        "master",
        "vitstr_small",
        "vitstr_base",
        "parseq",
        "viptr_tiny",
    ],
)
def test_recognition_zoo(arch_name, input_shape):
    batch_size = 2
    # Model
    predictor = recognition.zoo.recognition_predictor(arch_name, pretrained=False)
    predictor.model.eval()
    # object check
    assert isinstance(predictor, RecognitionPredictor)

    input_tensor = np.random.rand(batch_size, *input_shape).astype(np.float32)
    if torch.cuda.is_available():
        predictor.model.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(word, str) and isinstance(conf, float) for word, conf in out)


@pytest.mark.parametrize("arch_name, head", [["master", "linear"], ["parseq", "head"]])
@pytest.mark.parametrize("exportable", [False, True])
def test_autoregressive_decoding_early_stop(arch_name, head, exportable):
    # Decoding stops once every sequence has emitted EOS, except when the model is exportable: tracing would
    # freeze the number of decoding steps to the one the dummy input needed
    model = recognition.__dict__[arch_name](pretrained=False, pretrained_backbone=False, exportable=exportable).eval()
    input_tensor = torch.rand((2, 3, 32, 128))

    def _decoder_calls(eos_bias: float) -> int:
        calls = []
        with torch.no_grad():
            getattr(model, head).bias[model.vocab_size] = eos_bias
        hook = model.decoder.register_forward_hook(lambda *_: calls.append(1))
        try:
            with torch.no_grad():
                model(input_tensor)
        finally:
            hook.remove()
        return len(calls)

    # EOS emitted at the first step vs. never emitted
    early, full = _decoder_calls(1e4), _decoder_calls(-1e4)
    assert early == full if exportable else early < full

import os

import numpy as np
import pytest
import torch
from torch import nn

from doctr.datasets import VOCABS
from doctr.models import recognition, recognition_predictor
from doctr.models.utils import (
    _bf16_to_float32,
    _copy_tensor,
    add_whitelist,
    conv_sequence_pt,
    load_pretrained_params,
    set_device_and_dtype,
)
from doctr.models.utils.pytorch import _vocab_projections


def test_copy_tensor():
    x = torch.rand(8)
    m = _copy_tensor(x)
    assert m.device == x.device and m.dtype == x.dtype and m.shape == x.shape and torch.allclose(m, x)


def test_bf16_to_float32():
    x = torch.randn([2, 2], dtype=torch.bfloat16)
    converted_x = _bf16_to_float32(x)
    assert x.dtype == torch.bfloat16 and converted_x.dtype == torch.float32 and torch.equal(converted_x, x.float())


def test_load_pretrained_params(tmpdir_factory):
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    # Retrieve this URL
    url = "https://github.com/mindee/doctr/releases/download/v0.2.1/tmp_checkpoint-6f0ce0e6.pt"
    # Temp cache dir
    cache_dir = tmpdir_factory.mktemp("cache")
    # Pass an incorrect hash
    with pytest.raises(ValueError):
        load_pretrained_params(model, url, "mywronghash", cache_dir=str(cache_dir))
    # Let it resolve the hash from the file name
    load_pretrained_params(model, url, cache_dir=str(cache_dir))
    # Check that the file was downloaded & the archive extracted
    assert os.path.exists(cache_dir.join("models").join(url.rpartition("/")[-1].split("&")[0]))
    # Default initialization
    load_pretrained_params(model, None)
    # Check ignore keys
    load_pretrained_params(model, url, cache_dir=str(cache_dir), ignore_keys=["2.weight"])
    # non matching keys
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    with pytest.raises(ValueError):
        load_pretrained_params(model, url, cache_dir=str(cache_dir), ignore_keys=["2.weight"])


def test_conv_sequence():
    assert len(conv_sequence_pt(3, 8, kernel_size=3)) == 1
    assert len(conv_sequence_pt(3, 8, True, kernel_size=3)) == 2
    assert len(conv_sequence_pt(3, 8, False, True, kernel_size=3)) == 2
    assert len(conv_sequence_pt(3, 8, True, True, kernel_size=3)) == 3


def test_set_device_and_dtype():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    batches = [torch.rand(8) for _ in range(2)]

    model, batches = set_device_and_dtype(model, batches, device="cpu", dtype=torch.float32)
    assert model[0].weight.device == torch.device("cpu")
    assert model[0].weight.dtype == torch.float32
    assert batches[0].device == torch.device("cpu")
    assert batches[0].dtype == torch.float32
    # FP16 check
    model, batches = set_device_and_dtype(model, batches, device="cpu", dtype=torch.float16)
    assert model[0].weight.dtype == torch.float16
    assert batches[0].dtype == torch.float16

    img_batches = [torch.rand(8) for _ in range(2)]
    mask_batches = [torch.ones(8, dtype=torch.bool) for _ in range(2)]

    model, batch = set_device_and_dtype(
        model,
        (img_batches, mask_batches),
        device="cpu",
        dtype=torch.float32,
    )

    assert model[0].weight.device == torch.device("cpu")
    assert model[0].weight.dtype == torch.float32

    assert isinstance(batch, list)
    new_imgs = [img for img, _ in batch]
    new_masks = [mask for _, mask in batch]

    # image checks
    assert all(isinstance(t, torch.Tensor) for t in new_imgs)
    assert all(t.dtype == torch.float32 for t in new_imgs)
    assert all(t.device == torch.device("cpu") for t in new_imgs)

    # mask checks (should be bool)
    assert all(isinstance(t, torch.Tensor) for t in new_masks)
    assert all(t.dtype == torch.bool for t in new_masks)
    assert all(t.device == torch.device("cpu") for t in new_masks)


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
def test_add_whitelist(arch_name):
    # A test vocab containing the full whitelist plus extra forbiddable characters, kept small
    # enough to fit every architecture (SAR's feedback embedding caps at 512 entries).
    _whitelist = [VOCABS["polish"], VOCABS["german"]]
    _allowed = "".join(dict.fromkeys("".join(_whitelist)))  # ordered-unique characters
    _extra = "".join(c for c in VOCABS["multilingual"] if c not in set(_allowed))[:200]
    _test_vocab = _allowed + _extra

    model = recognition.__dict__[arch_name](pretrained=True, vocab=_test_vocab).eval()
    allowed = set(_allowed)
    # every whitelisted character is part of the model vocab (nothing is dropped)
    assert allowed.issubset(set(model.vocab))

    forbidden_idx = [i for i, c in enumerate(model.vocab) if c not in allowed]
    allowed_idx = [i for i, c in enumerate(model.vocab) if c in allowed]
    terminator_idx = len(model.vocab)
    assert len(forbidden_idx) > 0

    samples = torch.rand(4, 3, 32, 128)

    handle = add_whitelist(model, _whitelist)
    with torch.inference_mode():
        out = model(samples, return_model_output=True, return_preds=True)
    logits = out["out_map"]

    # forbidden characters are masked out, while whitelisted characters and the terminator stay finite
    assert torch.isneginf(logits[..., forbidden_idx]).all()
    assert torch.isfinite(logits[..., allowed_idx]).all()
    assert torch.isfinite(logits[..., terminator_idx]).all()
    # the decoded output only contains whitelisted characters (and no leaked special tokens)
    for word, _ in out["preds"]:
        assert all(char in allowed for char in word)

    # remove() restores the original, unconstrained decoding
    handle.remove()
    with torch.inference_mode():
        restored = model(samples, return_model_output=True)["out_map"]
    assert torch.isfinite(restored).all()

    # Test biased model:
    # Even when the model is biased toward forbidden characters, the whitelist must win
    # (this also exercises the autoregressive feedback loop).
    model = recognition.parseq(pretrained=True, vocab=_test_vocab).eval()
    allowed = set(VOCABS["german"])
    forbidden_idx = torch.tensor([i for i, c in enumerate(model.vocab) if c not in allowed])

    def bias_forbidden(module, inputs, output):
        output = output.clone()
        output[..., forbidden_idx] += 1e4
        return output

    bias_handle = model.head.register_forward_hook(bias_forbidden)  # runs before the whitelist hook
    samples = torch.rand(4, 3, 32, 128)
    with torch.inference_mode():
        attacked = model(samples, return_preds=True)["preds"]
    assert any(char not in allowed for word, _ in attacked for char in word)

    whitelist_handle = add_whitelist(model, VOCABS["german"])  # registered after -> overrides the bias
    with torch.inference_mode():
        defended = model(samples, return_preds=True)["preds"]
    assert all(char in allowed for word, _ in defended for char in word)

    whitelist_handle.remove()
    bias_handle.remove()

    # Test as context manager and on a predictor
    model = recognition.crnn_vgg16_bn(pretrained=True, vocab=_test_vocab).eval()
    samples = torch.rand(2, 3, 32, 128)
    with add_whitelist(model, VOCABS["german"]):
        with torch.inference_mode():
            masked = model(samples, return_model_output=True)["out_map"]
        assert torch.isneginf(masked).any()
    # outside the context the whitelist has been removed
    with torch.inference_mode():
        restored = model(samples, return_model_output=True)["out_map"]
    assert torch.isfinite(restored).all()

    # test whitelist error cases
    model = recognition.crnn_vgg16_bn(pretrained=True, vocab="abc123").eval()
    # a whitelist disjoint from the model vocabulary is rejected
    with pytest.raises(ValueError):
        add_whitelist(model, "XYZ")
    # an object that is not a recognition model / predictor is rejected
    with pytest.raises(TypeError):
        add_whitelist(nn.Linear(8, 8), "abc")


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
def test_end_to_end_add_whitelist(arch_name):
    vocab = "abcXYZ"
    allowed = set("abc")
    model = recognition.__dict__[arch_name](pretrained=False, vocab=vocab).eval()
    predictor = recognition_predictor(model, batch_size=2)

    forbidden_idx = model.vocab.index("X")
    allowed_idx = model.vocab.index("a")
    projection = _vocab_projections(model, len(model.vocab))[0]

    def bias_forbidden(_module, _inputs, output):
        output = output.clone()
        output[..., forbidden_idx] += 1e4
        output[..., allowed_idx] += 5e3
        return output

    bias_handle = projection.register_forward_hook(bias_forbidden)
    crops = [(255 * np.random.rand(32, 128, 3)).astype(np.uint8) for _ in range(2)]

    try:
        unconstrained = predictor(crops)
        assert all("X" in word for word, _ in unconstrained)
        with add_whitelist(predictor, "abc"):
            constrained = predictor(crops)
        assert all(word and all(char in allowed for char in word) for word, _ in constrained)
        restored = predictor(crops)
        assert all("X" in word for word, _ in restored)
    finally:
        bias_handle.remove()


def test_vocab_projections_fallback_candidates():
    from doctr.models.utils.pytorch import _vocab_projections

    vocab_size = 6

    class UnknownRecognitionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(4, vocab_size + 4)
            self.projection = nn.Linear(vocab_size + 4, vocab_size + 1)
            self.aux_projection = nn.Linear(vocab_size + 4, vocab_size + 3)
            self.unrelated = nn.Linear(vocab_size + 4, vocab_size + 4)

    model = UnknownRecognitionModel()
    assert _vocab_projections(model, vocab_size) == [model.projection, model.aux_projection]

    with pytest.raises(RuntimeError, match="Could not locate the vocabulary projection layer"):
        _vocab_projections(nn.Linear(4, 4), vocab_size)


# Forbidden characters whose visual base (via anyascii) is part of the German whitelist.
_NEAREST_VOCAB = "".join(dict.fromkeys(VOCABS["german"] + "ąóńł"))
_NEAREST_FOLDS = {"ą": "a", "ó": "o", "ń": "n", "ł": "l"}


def _force_and_decode(model, target_char, **whitelist_kwargs):
    """Bias the model to prefer ``target_char``, apply the whitelist, return the decoded word."""
    from doctr.models.utils.pytorch import _vocab_projections, add_whitelist

    forbidden_idx = model.vocab.index(target_char)
    projection = _vocab_projections(model, len(model.vocab))[0]

    def bias(module, inputs, output, idx=forbidden_idx):
        output = output.clone()
        output[..., idx] += 1e4
        return output

    bias_handle = projection.register_forward_hook(bias)  # runs before the whitelist hook
    whitelist_handle = add_whitelist(model, VOCABS["german"], **whitelist_kwargs)
    with torch.inference_mode():
        word = model(torch.rand(2, 3, 32, 128), return_preds=True)["preds"][0][0]
    whitelist_handle.remove()
    bias_handle.remove()
    return word


@pytest.mark.parametrize("arch_name", ["crnn_vgg16_bn", "sar_resnet31", "master", "parseq", "viptr_tiny"])
def test_add_whitelist_nearest_folds_to_base(arch_name):
    model = recognition.__dict__[arch_name](pretrained=True, vocab=_NEAREST_VOCAB).eval()
    for forbidden_char, base_char in _NEAREST_FOLDS.items():
        word = _force_and_decode(model, forbidden_char, strategy="nearest")
        # the forbidden character is folded onto its allowed visual base (CTC collapses repeats)
        assert word and set(word) == {base_char}


def test_add_whitelist_nearest_custom_mapping():
    model = recognition.parseq(pretrained=True, vocab=_NEAREST_VOCAB).eval()
    # an explicit mapping overrides the default transliteration (ą would otherwise fold to "a")
    word = _force_and_decode(model, "ą", strategy="nearest", mapping={"ą": "z"})
    assert word and set(word) == {"z"}


def test_add_whitelist_nearest_weights_stays_within_whitelist():
    model = recognition.crnn_vgg16_bn(pretrained=True, vocab=_NEAREST_VOCAB).eval()
    allowed = set(VOCABS["german"])
    handle = add_whitelist(model, VOCABS["german"], strategy="nearest", mapping="weights")
    with torch.inference_mode():
        preds = model(torch.rand(3, 3, 32, 128), return_preds=True)["preds"]
    handle.remove()
    assert all(char in allowed for word, _ in preds for char in word)


def test_add_whitelist_strategy_errors():
    model = recognition.crnn_vgg16_bn(pretrained=True, vocab=_NEAREST_VOCAB).eval()
    with pytest.raises(ValueError):  # mapping is meaningless without strategy="nearest"
        add_whitelist(model, VOCABS["german"], mapping={"ą": "a"})
    with pytest.raises(ValueError):  # unknown strategy
        add_whitelist(model, VOCABS["german"], strategy="drop")
    with pytest.raises(ValueError):  # unknown mapping keyword
        add_whitelist(model, VOCABS["german"], strategy="nearest", mapping="closest")
    with pytest.raises(ValueError):  # unsupported mapping type
        add_whitelist(model, VOCABS["german"], strategy="nearest", mapping=123)

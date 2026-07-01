# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from collections.abc import Iterable
from typing import Any

import torch
import validators
from anyascii import anyascii
from torch import nn

from doctr.utils.data import download_from_url

__all__ = [
    "load_pretrained_params",
    "conv_sequence_pt",
    "set_device_and_dtype",
    "export_model_to_onnx",
    "add_whitelist",
    "_copy_tensor",
    "_bf16_to_float32",
    "_CompiledModule",
]

# torch compiled model type
_CompiledModule = torch._dynamo.eval_frame.OptimizedModule


def _copy_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach()


def _bf16_to_float32(x: torch.Tensor) -> torch.Tensor:
    # bfloat16 is not supported in .numpy(): torch/csrc/utils/tensor_numpy.cpp:aten_to_numpy_dtype
    return x.float() if x.dtype == torch.bfloat16 else x


def load_pretrained_params(
    model: nn.Module,
    path_or_url: str | None = None,
    hash_prefix: str | None = None,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.pt")

    Args:
        model: the PyTorch model to be loaded
        path_or_url: the path or URL to the model parameters (checkpoint)
        hash_prefix: first characters of SHA256 expected hash
        ignore_keys: list of weights to be ignored from the state_dict
        **kwargs: additional arguments to be passed to `doctr.utils.data.download_from_url`
    """
    if path_or_url is None:
        logging.warning("No model URL or Path provided, using default initialization.")
        return

    archive_path = (
        download_from_url(path_or_url, hash_prefix=hash_prefix, cache_subdir="models", **kwargs)
        if validators.url(path_or_url)
        else path_or_url
    )

    # Read state_dict
    state_dict = torch.load(archive_path, map_location="cpu")

    # Remove weights from the state_dict
    if ignore_keys is not None and len(ignore_keys) > 0:
        for key in ignore_keys:
            if key in state_dict:
                state_dict.pop(key)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if any(k not in ignore_keys for k in missing_keys + unexpected_keys):
            raise ValueError(
                "Unable to load state_dict, due to non-matching keys.\n"
                + f"Unexpected keys: {unexpected_keys}\nMissing keys: {missing_keys}"
            )
    else:
        # Load weights
        model.load_state_dict(state_dict)


def conv_sequence_pt(
    in_channels: int,
    out_channels: int,
    act: bool = False,
    bn: bool = False,
    activation: nn.Module = nn.ReLU(inplace=True),
    **kwargs: Any,
) -> list[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        act: should an activation layer be added
        bn: should a batch normalization layer be added
        activation: the activation layer to be added if act is True
        **kwargs: additional arguments to be passed to the convolutional layer

    Returns:
        list of layers
    """
    # No bias before Batch norm
    kwargs["bias"] = kwargs.get("bias", not bn)
    # Add activation directly to the conv if there is no BN
    conv_seq: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, **kwargs)]

    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))

    if act:
        conv_seq.append(activation)

    return conv_seq


def set_device_and_dtype(
    model: Any,
    batches: list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]],
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[Any, list[torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]]:
    """Set the device and dtype of a model and its batches

    >>> import torch
    >>> from torch import nn
    >>> from doctr.models.utils import set_device_and_dtype
    >>> model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    >>> batches = [torch.rand(8) for _ in range(2)]
    >>> model, batches = set_device_and_dtype(model, batches, device="cuda", dtype=torch.float16)

    Args:
        model: the model to be set
        batches: the batches to be set
        device: the device to be used
        dtype: the dtype to be used

    Returns:
        the model and batches set
    """
    model = model.to(device=device, dtype=dtype)
    if isinstance(batches, tuple):
        return model, [
            (img.to(device=device, dtype=dtype), mask.to(device=device, dtype=torch.bool))
            for img, mask in zip(*batches)
        ]
    return model, [batch.to(device=device, dtype=dtype) for batch in batches]


def export_model_to_onnx(
    model: nn.Module,
    model_name: str,
    dummy_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    **kwargs: Any,
) -> str:
    """Export model to ONNX format.

    >>> import torch
    >>> from doctr.models.classification import resnet18
    >>> from doctr.models.utils import export_model_to_onnx
    >>> model = resnet18(pretrained=True)
    >>> export_model_to_onnx(model, "my_model", dummy_input=torch.randn(1, 3, 32, 32))

    Args:
        model: the PyTorch model to be exported
        model_name: the name for the exported model
        dummy_input: the dummy input to the model
        input_names: optional names for the model inputs. Defaults to ``["input"]`` (or ``["input", "masks"]``
            when ``dummy_input`` is a tuple).
        output_names: optional names for the model outputs. Defaults to ``["logits"]`` (or
            ``["logits", "pred_boxes"]`` when ``dummy_input`` is a tuple). Pass the names of every output when
            the model returns more than one tensor (e.g. a multi-head model).
        dynamic_axes: optional dynamic axes. Defaults to a dynamic batch dimension on every input and output.
        kwargs: additional arguments to be passed to torch.onnx.export

    Returns:
        the path to the exported model
    """
    is_tuple = isinstance(dummy_input, tuple)
    if input_names is None:
        input_names = ["input", "masks"] if is_tuple else ["input"]
    if output_names is None:
        output_names = ["logits", "pred_boxes"] if is_tuple else ["logits"]
    if dynamic_axes is None:
        dynamic_axes = {name: {0: "batch_size"} for name in [*input_names, *output_names]}

    torch.onnx.export(
        model,
        dummy_input,  # type: ignore[arg-type]
        f"{model_name}.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True,
        dynamo=False,
        verbose=False,
        **kwargs,
    )
    logging.info(f"Model exported to {model_name}.onnx")
    return f"{model_name}.onnx"


# Location of the final vocabulary-projection layer for each recognition architecture.
_RECOGNITION_PROJECTIONS: dict[str, str] = {
    "CRNN": "linear",
    "SAR": "decoder.output_dense",
    "MASTER": "linear",
    "ViTSTR": "head",
    "PARSeq": "head",
    "VIPTR": "head",
}


class WhitelistHandle:
    """Removable registration returned by :func:`add_whitelist`.

    Call :meth:`remove` to restore the model's original, unconstrained decoding. The
    handle can also be used as a context manager, in which case the whitelist is removed
    on exit.
    """

    def __init__(self, handles: list[torch.utils.hooks.RemovableHandle]) -> None:
        self._handles = handles

    def remove(self) -> None:
        """Remove the whitelist and restore the model's unconstrained decoding."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "WhitelistHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove()


def _recognition_model(model: nn.Module) -> nn.Module:
    # Accept an ocr_predictor / kie_predictor / recognition_predictor or a recognition model
    if hasattr(model, "vocab") and hasattr(model, "postprocessor"):
        return model
    reco_predictor = getattr(model, "reco_predictor", model)
    reco_model = getattr(reco_predictor, "model", None)
    if reco_model is None:
        raise TypeError(
            "Expected an ocr_predictor, kie_predictor, recognition_predictor or a recognition "
            f"model, but could not find a recognition model on {type(model).__name__}."
        )
    return reco_model


def _vocab_projections(model: nn.Module, vocab_size: int) -> list[nn.Linear]:
    path = _RECOGNITION_PROJECTIONS.get(type(model).__name__)
    if path is not None:
        layer: Any = model
        for part in path.split("."):
            layer = getattr(layer, part)
        if isinstance(layer, nn.Linear):
            return [layer]
    # Fallback for unknown architectures: any Linear projecting to the vocab (+ up to 3 specials)
    candidates = [
        module
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.out_features in {vocab_size + 1, vocab_size + 2, vocab_size + 3}
    ]
    if not candidates:
        raise RuntimeError(f"Could not locate the vocabulary projection layer of {type(model).__name__}.")
    return candidates


def _anyascii_nearest_map(vocab: str, allowed: set[str]) -> dict[str, str]:
    """Map each forbidden character to the visually closest allowed one via transliteration.

    Uses ``anyascii`` to fold characters to their ASCII form (e.g. ``ä -> a``, ``ł -> l``,
    Cyrillic ``а -> a``); a forbidden character is mapped to an allowed character sharing the
    same ASCII form. Forbidden characters without such a match are left unmapped (they fall
    back to plain masking).
    """
    by_translit: dict[str, str] = {}
    for char in vocab:
        if char not in allowed:
            continue
        key = anyascii(char)
        current = by_translit.get(key)
        # Prefer a pure-ASCII allowed character as the canonical target for a given form.
        if current is None or (char == anyascii(char) and current != anyascii(current)):
            by_translit[key] = char

    mapping: dict[str, str] = {}
    for char in vocab:
        if char in allowed:
            continue
        form = anyascii(char)
        target = by_translit.get(form) or by_translit.get(form.lower()) or by_translit.get(form[:1])
        if target is not None:
            mapping[char] = target
    return mapping


def _weights_nearest_map(vocab: str, allowed: set[str], projection: nn.Linear) -> dict[str, str]:
    """Map each forbidden character to the allowed one whose projection weights are most similar.

    This uses the model's own learned representation: the nearest allowed character is the one
    the model most confuses the forbidden character with (cosine similarity of the projection
    weight rows).
    """
    vocab_size = len(vocab)
    rows = nn.functional.normalize(projection.weight.detach()[:vocab_size], dim=1)
    allowed_idx = [i for i, char in enumerate(vocab) if char in allowed]
    forbidden_idx = [i for i, char in enumerate(vocab) if char not in allowed]
    if not allowed_idx or not forbidden_idx:
        return {}
    similarity = rows[forbidden_idx] @ rows[allowed_idx].t()
    nearest = similarity.argmax(dim=1)
    return {vocab[forbidden_idx[k]]: vocab[allowed_idx[int(nearest[k])]] for k in range(len(forbidden_idx))}


def _keep_and_reassign(
    vocab: str, allowed: set[str], out_features: int, char_map: dict[str, str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the keep mask and the (forbidden -> allowed) index tensors for one projection."""
    vocab_size = len(vocab)
    keep = torch.zeros(out_features, dtype=torch.bool)
    for idx, char in enumerate(vocab):
        keep[idx] = char in allowed
    keep[vocab_size] = True  # sequence terminator (CTC blank / attention <eos>)

    position = {char: idx for idx, char in enumerate(vocab)}
    src, dst = [], []
    for forbidden_char, allowed_char in char_map.items():
        src_idx, dst_idx = position.get(forbidden_char), position.get(allowed_char)
        # only reassign genuinely-forbidden characters onto genuinely-allowed ones
        if src_idx is not None and dst_idx is not None and not keep[src_idx] and allowed_char in allowed:
            src.append(src_idx)
            dst.append(dst_idx)
    return keep, torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


def add_whitelist(
    model: nn.Module,
    vocabs: str | Iterable[str],
    *,
    strategy: str = "mask",
    mapping: str | dict[str, str] | None = None,
    verbose: bool = False,
) -> WhitelistHandle:
    """Restrict a recognition model so it can only predict a subset of its vocabulary.

    The whitelist is enforced at the model's final projection layer, before the decoding
    ``argmax``. Because the projection is the single point every logit flows through, the
    constraint also applies inside the autoregressive decoding loop of SAR, MASTER and PARSeq,
    so a forbidden character can never be produced -- not even fed back mid-word. The sequence
    terminator (CTC ``blank`` / attention ``<eos>``) is always kept so decoding still
    terminates. It works with every recognition architecture and with any predictor wrapping
    one (`ocr_predictor`, `kie_predictor`, `recognition_predictor`).

    Two strategies are available:

    * ``"mask"`` (default): the logits of forbidden characters are set to ``-inf``, so decoding
      falls back to the highest-scoring allowed character.
    * ``"nearest"``: the score of each forbidden character is first reassigned to the closest
      allowed character (so e.g. ``ä`` folds onto ``a``), then forbidden logits are masked.
      Forbidden characters without a mapping fall back to masking.

    A whitelist can only restrict a model to characters it already knows: characters that are
    not part of the model's own vocabulary are silently ignored.

    >>> from doctr.datasets import VOCABS
    >>> from doctr.models import ocr_predictor
    >>> from doctr.models.utils import add_whitelist
    >>> predictor = ocr_predictor(pretrained=True)
    >>> handle = add_whitelist(predictor, [VOCABS["polish"], VOCABS["german"]])
    >>> # ... run the predictor; only Polish/German characters can be predicted ...
    >>> handle.remove()  # restore the original, unconstrained decoding

    Args:
        model: an `ocr_predictor`, `kie_predictor`, `recognition_predictor`, or a recognition model.
        vocabs: a vocabulary string (e.g. ``VOCABS["german"]``) or an iterable of vocabulary
            strings (e.g. ``[VOCABS["polish"], VOCABS["german"]]``) whose characters are allowed.
        strategy: ``"mask"`` (default) to drop forbidden characters, or ``"nearest"`` to fold
            them onto the closest allowed character.
        mapping: only used when ``strategy="nearest"``. ``None`` or ``"anyascii"`` builds the
            forbidden-to-allowed map by transliteration (the default); ``"weights"`` derives it
            from the projection weights (the model's own confusions); a ``dict`` of
            ``{forbidden_char: allowed_char}`` overrides specific characters on top of the
            transliteration map.
        verbose: if True, log how many characters were kept, forbidden and reassigned per model.

    Returns:
        a :class:`WhitelistHandle`; call its :meth:`~WhitelistHandle.remove` method to restore
        the original, unconstrained decoding.
    """
    if strategy not in {"mask", "nearest"}:
        raise ValueError(f"Unknown strategy {strategy!r}; expected 'mask' or 'nearest'.")
    if strategy == "mask" and mapping is not None:
        raise ValueError("The 'mapping' argument is only used with strategy='nearest'.")
    if isinstance(mapping, str) and mapping not in {"anyascii", "weights"}:
        raise ValueError(f"Unknown mapping {mapping!r}; expected 'anyascii', 'weights', a dict or None.")
    if mapping is not None and not isinstance(mapping, (str, dict)):
        raise ValueError("The 'mapping' argument must be None, 'anyascii', 'weights' or a dict.")

    allowed = set(vocabs) if isinstance(vocabs, str) else {char for vocab in vocabs for char in vocab}

    handles: list[torch.utils.hooks.RemovableHandle] = []
    reco_model = _recognition_model(model)
    vocab: str = reco_model.vocab  # type: ignore[assignment]
    vocab_size = len(vocab)
    if not any(char in allowed for char in vocab):
        raise ValueError(
            "The whitelist shares no character with the model's vocabulary; the model would "
            "be unable to predict anything."
        )

    # A vocab-level character map (shared by every projection); the weight-based map is
    # derived per projection further down.
    base_map: dict[str, str] = {}
    if strategy == "nearest" and mapping != "weights":
        base_map = _anyascii_nearest_map(vocab, allowed)
        if isinstance(mapping, dict):
            base_map = {**base_map, **mapping}

    reassigned = 0
    for projection in _vocab_projections(reco_model, vocab_size):
        char_map = (
            _weights_nearest_map(vocab, allowed, projection)
            if strategy == "nearest" and mapping == "weights"
            else base_map
        )
        keep, src, dst = _keep_and_reassign(vocab, allowed, projection.out_features, char_map)
        reassigned = max(reassigned, src.numel())

        def _constrain_logits(
            _module: nn.Module,
            _inputs: Any,
            output: torch.Tensor,
            keep: torch.Tensor = keep,
            src: torch.Tensor = src,
            dst: torch.Tensor = dst,
        ):
            output = output.clone()
            if src.numel():
                # move each forbidden character's score onto its nearest allowed character
                values = output[..., src.to(output.device)]
                index = dst.to(output.device).view(*([1] * (output.dim() - 1)), -1).expand(*output.shape[:-1], -1)
                output.scatter_reduce_(-1, index, values, reduce="amax", include_self=True)
            output[..., ~keep.to(output.device)] = float("-inf")
            return output

        handles.append(projection.register_forward_hook(_constrain_logits))

    if verbose:  # pragma: no cover
        kept = sum(char in allowed for char in vocab)
        logging.info(
            f"add_whitelist: {type(reco_model).__name__} - kept {kept}/{vocab_size} vocabulary "
            f"characters, forbade {vocab_size - kept}"
            + (f", reassigned {reassigned} to a nearest allowed character." if strategy == "nearest" else ".")
        )

    return WhitelistHandle(handles)

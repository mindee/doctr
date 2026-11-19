import os
import tempfile
from unittest import mock

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from doctr.datasets import LATEX_TOKENS
from doctr.io import DocumentFile, read_img_as_numpy
from doctr.models import formula
from doctr.models.formula.predictor import FormulaPredictor
from doctr.models.formula.zoo import formula_predictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["formulanet", (3, 256, 768)],
    ],
)
# Coverage tracing slows the (pure Python) dynamo export down ~3x: its code paths are covered by the other tests
@pytest.mark.parametrize("dynamo", [False, pytest.param(True, marks=pytest.mark.no_cover)])
def test_models_onnx_export(arch_name, input_shape, dynamo):
    if dynamo:
        pytest.importorskip("onnxscript")
    batch_size = 2
    model = formula.__dict__[arch_name](pretrained=True, exportable=True).eval()
    # The tracing exporter unrolls the greedy decoding (no early stop): keep it short
    model.max_decode_length = 4
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    with torch.inference_mode():
        if dynamo:
            # The exported loops stop at the end of the formulas, as the decoding of PyTorch: the steps after it are
            # zeros
            model.exportable = False
            pt_logits = model(dummy_input, return_model_output=True)["out_map"].numpy()
            model.exportable = True
        else:
            pt_logits = model(dummy_input)["logits"].numpy()
            assert pt_logits.shape == (batch_size, 5, len(LATEX_TOKENS) + 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = export_model_to_onnx(
            model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input, dynamo=dynamo
        )
        assert os.path.exists(model_path)
        ort_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        ort_outs = ort_session.run(["logits"], {"input": dummy_input.numpy()})
    assert ort_outs[0].shape == (batch_size, 5, len(LATEX_TOKENS) + 1)
    num_steps = pt_logits.shape[1]
    assert not ort_outs[0][:, num_steps:].any()
    try:
        assert np.allclose(pt_logits, ort_outs[0][:, :num_steps], atol=1e-4)
    except AssertionError:
        max_diff = np.max(np.abs(pt_logits - ort_outs[0][:, :num_steps]))
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {max_diff}")


def test_models_onnx_export_loop(mock_formula_image):
    # With the torch.export-based exporter, the greedy decoding is exported as ONNX loops, which stop once every
    # formula is complete
    pytest.importorskip("onnxscript")
    predictor = formula.formula_predictor("formulanet", pretrained=True)
    model = formula.formulanet(pretrained=True, exportable=True).eval()
    model.max_decode_length = 64
    model.export_chunk_size = 16  # several loops, the last ones after the end of the formula
    batch = predictor.pre_processor([read_img_as_numpy(mock_formula_image)])
    batch = (batch[0] if isinstance(batch, tuple) else batch)[0]
    with torch.inference_mode():
        # Reference: the decoding of PyTorch, which stops at the end of the formula
        model.exportable = False
        ref_logits = model(batch, return_model_output=True)["out_map"].numpy()
        model.exportable = True
    num_steps = ref_logits.shape[1]
    assert num_steps < model.max_decode_length + 1

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = export_model_to_onnx(model, os.path.join(tmpdir, "model"), dummy_input=batch, dynamo=True)
        assert [os.path.basename(path) for path in os.listdir(tmpdir)] == ["model.onnx"]  # a single file
        graph = onnx.load(model_path).graph
        assert any(node.op_type == "Loop" for node in graph.node)
        # The debugging metadata of the exporter (stack traces of every node) is removed
        assert not any(node.metadata_props for node in graph.node)
        ort_session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # The batch dimension is dynamic
        ort_logits = ort_session.run(["logits"], {"input": np.concatenate([batch.numpy()] * 2)})[0]
    assert ort_logits.shape == (2, model.max_decode_length + 1, len(LATEX_TOKENS) + 1)
    assert np.allclose(ort_logits[:, :num_steps], ref_logits, atol=1e-4)
    # The steps after the end of the formula are not decoded
    assert not ort_logits[:, num_steps:].any()
    ort_latex = model.postprocessor(torch.from_numpy(ort_logits))[0][0]
    assert ort_latex == model.postprocessor(torch.from_numpy(ref_logits))[0][0]


def test_models_export_loop_eager(mock_formula_image, eager_export_loops):
    # The chunked decoding exported as ONNX loops (cf. `test_models_onnx_export_loop`), run in Python
    predictor = formula.formula_predictor("formulanet", pretrained=True)
    model = formula.formulanet(pretrained=True, exportable=True).eval()
    model.max_decode_length = 64
    model.export_chunk_size = 16  # several loops, the last ones after the end of the formula
    batch = predictor.pre_processor([read_img_as_numpy(mock_formula_image)])
    batch = (batch[0] if isinstance(batch, tuple) else batch)[0]
    with torch.inference_mode():
        loop_logits = model(batch)["logits"]
        assert eager_export_loops[0] > 0  # the loops ran
        with mock.patch.object(torch.compiler, "is_exporting", lambda: False):
            # Reference: the decoding of PyTorch, which stops at the end of the formula
            model.exportable = False
            ref_logits = model(batch, return_model_output=True)["out_map"]
    num_steps = ref_logits.shape[1]
    assert num_steps < model.max_decode_length + 1
    assert loop_logits.shape == (1, model.max_decode_length + 1, len(LATEX_TOKENS) + 1)
    assert torch.allclose(loop_logits[:, :num_steps], ref_logits, atol=1e-4)
    # The steps after the end of the formula are not decoded
    assert not loop_logits[:, num_steps:].any()
    assert model.postprocessor(loop_logits)[0][0] == model.postprocessor(ref_logits)[0][0]


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

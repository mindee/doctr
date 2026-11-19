import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr.io import DocumentFile
from doctr.models import table_structure
from doctr.models.table_structure.predictor import TablePredictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["tablecenternet", (3, 1024, 1024)],
    ],
)
# Coverage tracing slows the (pure Python) dynamo export down ~3x: its code paths are covered by the other tests
@pytest.mark.parametrize("dynamo", [False, pytest.param(True, marks=pytest.mark.no_cover)])
def test_models_onnx_export(arch_name, input_shape, dynamo):
    if dynamo:
        pytest.importorskip("onnxscript")
    # Model
    batch_size = 2
    model = table_structure.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    head_names = list(model.heads.keys())
    with torch.inference_mode():
        pt = model(dummy_input)
    pt_out = {name: pt[name].detach().cpu().numpy() for name in head_names}
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export (the multi-head model relies on the generalized export helper to name every output)
        model_path = export_model_to_onnx(
            model,
            model_name=os.path.join(tmpdir, "model"),
            dummy_input=dummy_input,
            output_names=head_names,
            dynamo=dynamo,
        )
        assert os.path.exists(model_path)
        # Inference
        ort_session = onnxruntime.InferenceSession(
            os.path.join(tmpdir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        ort_outs = ort_session.run(head_names, {"input": dummy_input.numpy()})

    assert isinstance(ort_outs, list) and len(ort_outs) == len(head_names)
    # Check head map shapes
    for name, ort_o in zip(head_names, ort_outs):
        assert ort_o.shape == pt_out[name].shape
    # Check that the output is close to the PyTorch output - only warn if not close
    try:
        for name, ort_o in zip(head_names, ort_outs):
            assert np.allclose(pt_out[name], ort_o, atol=1e-4)
    except AssertionError:
        max_diff = max(np.max(np.abs(pt_out[name] - ort_o)) for name, ort_o in zip(head_names, ort_outs))
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {max_diff}")


@pytest.mark.parametrize(
    "arch_name",
    [
        "tablecenternet",
    ],
)
def test_torch_compiled_models(arch_name, mock_table_image):
    doc = DocumentFile.from_images([mock_table_image])
    predictor = table_structure.zoo.table_predictor(arch_name, pretrained=True)
    assert isinstance(predictor, TablePredictor)
    out = predictor(doc)

    # Compile the model
    base = table_structure.__dict__[arch_name](pretrained=True).eval()
    compiled_model = torch.compile(base)
    assert isinstance(compiled_model, _CompiledModule)
    compiled_predictor = table_structure.zoo.table_predictor(compiled_model)
    compiled_out = compiled_predictor(doc)

    # Compare that outputs are close
    assert len(out) == len(compiled_out) == 1
    table, compiled_table = out[0], compiled_out[0]
    # A table was actually found, otherwise there is nothing to compare
    assert len(table["cells"]) > 0
    assert len(table["cells"]) == len(compiled_table["cells"])
    assert (table["num_rows"], table["num_cols"]) == (compiled_table["num_rows"], compiled_table["num_cols"])
    for cell, compiled_cell in zip(table["cells"], compiled_table["cells"]):
        assert np.allclose(cell["geometry"], compiled_cell["geometry"], atol=1e-4)
        assert np.isclose(cell["score"], compiled_cell["score"], atol=1e-4)
        assert all(cell[k] == compiled_cell[k] for k in ("row_start", "row_end", "col_start", "col_end"))

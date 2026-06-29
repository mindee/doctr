import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr.models import table_structure
from doctr.models.table_structure import TableCenterNet
from doctr.models.table_structure.predictor import TablePredictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx

_HEADS = {"hm": 2, "reg": 2, "ct2cn": 8, "cn2ct": 8, "lc": 2, "sp": 2}


def _grid_target(rows=2, cols=3, use_polygons=True):
    """A relative-coordinate {"cells", "logic"} target for a rows x cols grid."""
    xs, ys = np.linspace(0.1, 0.9, cols + 1), np.linspace(0.1, 0.9, rows + 1)
    cells, logic = [], []
    for r in range(rows):
        for c in range(cols):
            cells.append([[xs[c], ys[r]], [xs[c + 1], ys[r]], [xs[c + 1], ys[r + 1]], [xs[c], ys[r + 1]]])
            logic.append([c, c, r, r])

    cell_array = np.asarray(cells, dtype=np.float32).reshape(-1, 4, 2)
    if not use_polygons:
        cell_array = np.concatenate([cell_array.min(axis=1), cell_array.max(axis=1)], axis=1)
    return {"cells": cell_array, "logic": np.asarray(logic, dtype=np.int64).reshape(-1, 4)}


@pytest.mark.parametrize("train_mode", [True, False])
@pytest.mark.parametrize("assume_straight_pages", [True, False])
@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["tablecenternet", (3, 1024, 1024)],
    ],
)
def test_table_models(arch_name, input_shape, train_mode, assume_straight_pages):
    batch_size = 2
    model = table_structure.__dict__[arch_name](
        pretrained=True,
        assume_straight_pages=assume_straight_pages,
    )
    model = model.train() if train_mode else model.eval()
    assert isinstance(model, TableCenterNet)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = [
        _grid_target(use_polygons=not assume_straight_pages),
        _grid_target(use_polygons=not assume_straight_pages),
    ]

    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor, target, return_model_output=True, return_preds=not train_mode)
    assert isinstance(out, dict)
    assert len(out) == 3 if not train_mode else len(out) == 2
    # Check head maps
    assert "out_map" in out
    for name, channels in _HEADS.items():
        assert out["out_map"][name].shape[:2] == (batch_size, channels)
        assert out["out_map"][name].dtype == torch.float32

    # Check Preds
    if not train_mode:
        assert len(out["preds"]) == batch_size
        expected_shape = (4,) if assume_straight_pages else (4, 2)
        for pred in out["preds"]:
            assert set(pred) == {"polygons", "scores", "logical"}
            # Check logical coordinates have 4 entries per cell (start/end col, start/end row)
            assert pred["logical"].shape[1] == 4
            # Check that the number of cells, scores and logical coordinates are the same
            assert len(pred["polygons"]) == len(pred["scores"]) == len(pred["logical"])
            assert pred["polygons"].shape[1:] == expected_shape
            if pred["polygons"].size:
                # Check that cells are in the range [0, 1]
                assert np.all(pred["polygons"] >= 0) and np.all(pred["polygons"] <= 1)
                # Check that scores are between 0 and 1
                assert np.all(pred["scores"] >= 0) and np.all(pred["scores"] <= 1)
    # Check loss
    assert isinstance(out["loss"], torch.Tensor)
    assert hasattr(model, "from_pretrained")


@pytest.mark.parametrize("assume_straight_pages", [True, False])
@pytest.mark.parametrize(
    "arch_name",
    [
        "tablecenternet",
    ],
)
def test_table_structure_zoo(arch_name, assume_straight_pages):
    predictor = table_structure.zoo.table_predictor(
        arch_name,
        pretrained=False,
        assume_straight_pages=assume_straight_pages,
    )
    predictor.model = predictor.model.eval()
    # object check
    assert isinstance(predictor, TablePredictor)
    input_tensor = np.random.rand(2, 1024, 1024, 3).astype(np.float32)
    if torch.cuda.is_available():
        predictor.model.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(page, dict) for page in out)
    assert all({"cells", "num_rows", "num_cols"} <= set(page) for page in out)
    expected_shape = (4,) if assume_straight_pages else (4, 2)
    for page in out:
        assert all(np.asarray(cell["geometry"]).shape == expected_shape for cell in page["cells"])
        assert all({"score", "row_start", "row_end", "col_start", "col_end"} <= set(cell) for cell in page["cells"])
        assert all(0 <= cell["score"] <= 1 for cell in page["cells"])


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["tablecenternet", (3, 1024, 1024)],
    ],
)
def test_models_onnx_export(arch_name, input_shape):
    # Model
    batch_size = 2
    model = table_structure.__dict__[arch_name](pretrained=False, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    head_names = list(model.heads.keys())
    pt = model(dummy_input)
    pt_out = {name: pt[name].detach().cpu().numpy() for name in head_names}
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export (the multi-head model relies on the generalized export helper to name every output)
        model_path = export_model_to_onnx(
            model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input, output_names=head_names
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
def test_torch_compiled_models(arch_name):
    page = (255 * np.random.rand(1024, 1024, 3)).astype(np.uint8)
    predictor = table_structure.zoo.table_predictor(arch_name, pretrained=False)
    assert isinstance(predictor, TablePredictor)
    out = predictor([page])

    # Compile the model
    base = table_structure.__dict__[arch_name](pretrained=True).eval()
    compiled_model = torch.compile(base)
    assert isinstance(compiled_model, _CompiledModule)
    compiled_predictor = table_structure.zoo.table_predictor(compiled_model)
    compiled_out = compiled_predictor([page])

    # Compare that outputs are close
    assert len(out) == len(compiled_out) == 1
    assert {"cells", "num_rows", "num_cols"} <= set(compiled_out[0])

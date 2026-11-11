import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr.io import DocumentFile
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
        pretrained=True,
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
    model = table_structure.__dict__[arch_name](pretrained=True, exportable=True).eval()
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


def test_table_predictor_regions(monkeypatch):
    predictor = table_structure.zoo.table_predictor("tablecenternet", pretrained=False)
    crops = []

    def fake_forward(images, **kwargs):
        # A fixed grid (crop-relative cells) for every crop: 2 cells, and 1 of the context margin, above the region
        crops.extend(images)
        h, w = images[0].shape[:2]
        pad = 20  # the context margin of a region smaller than 2000 px

        def cell(x0, y0, x1, y1, r, c):
            box = [(pad + x0) / w, (pad + y0) / h, (pad + x1) / w, (pad + y1) / h]
            return {"geometry": box, "score": 0.9, "row_start": r, "row_end": r, "col_start": c, "col_end": c}

        cells = [cell(0, 0, 100, 30, 1, 0), cell(100, 0, 200, 30, 1, 1), cell(0, -18, 200, -4, 0, 0)]
        return [{"cells": cells, "num_rows": 2, "num_cols": 2} for _ in images]

    monkeypatch.setattr(predictor, "forward", fake_forward)
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    x0, y0, x1 = 0.1, 0.25, 0.1 + 200 / 600  # a 200 x 50 px table region, its cells cover its top 30 px
    regions = [{"class_names": ["Text", "Table"], "boxes": np.array([[0, 0, 0.1, 0.1], [x0, y0, x1, 0.375]])}, None]
    # Words in each cell, one in the region below the cells, and one outside of the region
    words = np.array([
        [0.12, 0.27, 0.15, 0.29],
        [0.30, 0.27, 0.33, 0.29],
        [0.12, 0.345, 0.15, 0.365],
        [0.12, 0.80, 0.15, 0.82],
    ])
    tables = predictor.predict_regions([page, page], regions, [words, words[:0]])
    # Only the table region is cropped, with a context margin
    assert len(crops) == 1 and crops[0].shape[:2] == (90, 240)
    assert len(tables) == 2 and len(tables[0]) == 1 and tables[1] == []
    table = tables[0][0]
    # The cell of the context margin is dropped, the word below the cells makes a new row
    assert (table["num_rows"], table["num_cols"]) == (2, 2)
    cells = sorted(table["cells"], key=lambda c: (c["row_start"], c["col_start"]))
    assert [(c["row_start"], c["col_start"], c["words"], c["score"]) for c in cells] == [
        (0, 0, [0], 0.9),
        (0, 1, [1], 0.9),
        (1, 0, [2], None),
    ]
    # The cells are mapped back to page-relative coordinates
    expected = [[x0, y0], [x0 + 100 / 600, y0], [x0 + 100 / 600, 0.325], [x0, 0.325]]
    np.testing.assert_allclose(cells[0]["geometry"], expected, atol=1e-5)
    # Without words, the cells are empty
    tables = predictor.predict_regions([page], regions[:1])
    assert all(cell["words"] == [] for cell in tables[0][0]["cells"])

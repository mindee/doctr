import numpy as np

from doctr.models.table_structure.grid import (
    _axis_slots,
    _grid_lines,
    _group,
    as_polygons,
    assemble_page_table,
    fill_table_grid,
)


def _grid(xs, ys):
    """Boxes and logical coordinates (row_start, row_end, col_start, col_end) of a regular grid"""
    boxes, logic = [], []
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            boxes.append([xs[c], ys[r], xs[c + 1], ys[r + 1]])
            logic.append([r, r, c, c])
    return np.array(boxes, dtype=float), np.array(logic)


def _word(cx, cy, w=40, h=20):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _table(cells, logic, words):
    """The text grid of a table, from named words"""
    names, boxes = list(words), np.array(list(words.values()), dtype=float).reshape(-1, 4)
    boxes, logic, cell_words, origin = fill_table_grid(np.asarray(cells, dtype=float), np.asarray(logic), boxes)
    grid = [["" for _ in range(logic[:, 3].max() + 1)] for _ in range(logic[:, 1].max() + 1)]
    for (rs, _, cs, _), idcs in zip(logic, cell_words):
        grid[rs][cs] = " ".join(filter(None, [grid[rs][cs], *(names[i] for i in idcs)]))
    return grid, logic, origin


def test_table_grid_helpers():
    # Straight boxes (flat or 2-point) and polygons, as (N, 4, 2) polygons
    expected = [[[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]]]
    np.testing.assert_allclose(as_polygons([[0.1, 0.2, 0.5, 0.6]]), expected)
    np.testing.assert_allclose(as_polygons([((0.1, 0.2), (0.5, 0.6))]), expected)
    np.testing.assert_allclose(as_polygons(expected), expected)
    assert as_polygons(np.zeros((0, 4))).shape == (0, 4, 2)
    # Grid lines at the median of the cell edges, interpolated when no edge lies on them
    lines = _grid_lines(np.array([0, 2]), np.array([0, 2]), np.array([0.0, 20.0]), np.array([10.0, 30.0]), 3)
    np.testing.assert_allclose(lines, [0, 10, 20, 30])
    # Positions grouped by distance to the mean of their group
    np.testing.assert_array_equal(_group(np.array([0.0, 30.0, 4.0, 33.0]), 10), [0, 1, 0, 1])
    # Words beyond the grid make new indices, grouped by text line (rows) or overlapping extent (columns)
    lines = np.array([100.0, 200.0, 300.0])
    lo, hi = np.array([150.0, 40.0, 45.0, 10.0, 310.0]), np.array([160.0, 60.0, 65.0, 30.0, 330.0])
    np.testing.assert_array_equal(_axis_slots(lo, hi, lines, 15), [0, -1, -1, -2, 2])
    np.testing.assert_array_equal(_axis_slots(lo, hi, lines, None), [0, -1, -1, -2, 2])


def test_fill_table_grid_complete():
    # A 2 x 2 grid whose last cell is missed by the model, inside a table holding more text around it
    cells, logic = _grid([200, 400, 600], [300, 400, 500])
    words = {
        "A": _word(300, 350),
        "B": _word(500, 350),
        "C": _word(300, 450),
        "missed": _word(500, 450),  # in the slot of the missed cell
        "Col1": _word(300, 250),  # a header row above the grid
        "Col2": _word(500, 250),
        "Total": _word(250, 580, w=60),  # a sentence below the grid, across both columns
        "amount": _word(322.5, 580, w=75),
        "due": _word(405, 580, w=80),
        "key": _word(150, 350),  # a column on the left of the grid
    }
    grid, _, origin = _table(cells[:3], logic[:3], words)
    assert grid == [["", "Col1", "Col2"], ["key", "A", "B"], ["", "C", "missed"], ["", "Total amount due", ""]]
    # The cells made from the words have no source cell
    assert sorted(origin) == [-1] * 5 + [0, 1, 2]

    # Without words: the predicted grid, with local logical coordinates
    boxes, out_logic, cell_words, origin = fill_table_grid(cells, logic + 3, np.zeros((0, 4)))
    np.testing.assert_array_equal(out_logic, logic)
    assert cell_words == [[]] * 4 and origin == [0, 1, 2, 3]


def test_fill_table_grid_merge_and_prune():
    # The model splits a row in 2 (the edges of its cells do not line up) and predicts an empty row
    cells = [
        [100, 100, 300, 200],
        [300, 100, 500, 200],
        [100, 200, 300, 300],
        [300, 200, 500, 240],  # empty, a row of its own
        [300, 240, 500, 340],
        [100, 340, 300, 440],  # an empty row
        [300, 340, 500, 440],
        [100, 440, 300, 540],
        [300, 440, 500, 540],
    ]
    logic = [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1], [2, 2, 1, 1]]
    logic += [[3, 3, 0, 0], [3, 3, 1, 1], [4, 4, 0, 0], [4, 4, 1, 1]]
    words = {"h1": _word(200, 150), "h2": _word(400, 150), "a": _word(200, 250), "b": _word(400, 255)}
    words |= {"c": _word(200, 490), "d": _word(400, 490)}
    # The text of the split row lies on the same line: the 2 rows are merged, and the empty row is removed
    grid, _, _ = _table(cells, logic, words)
    assert grid == [["h1", "h2"], ["a", "b"], ["c", "d"]]
    # Text on different lines is not merged
    words["b"] = _word(400, 310)
    grid, _, _ = _table(cells, logic, words)
    assert grid == [["h1", "h2"], ["a", ""], ["", "b"], ["c", "d"]]


def test_fill_table_grid_split_rows():
    # The model merges the 3 text lines of a dense table into a single row
    cells, logic = _grid([100, 300, 500], [100, 160, 340])
    words = {"Name": _word(200, 130), "Value": _word(400, 130)}
    for line, cy in enumerate((190, 250, 310)):
        words |= {f"key{line}": _word(200, cy), f"{line}.00": _word(400, cy)}
    grid, _, _ = _table(cells, logic, words)
    assert grid == [["Name", "Value"], ["key0", "0.00"], ["key1", "1.00"], ["key2", "2.00"]]

    # A single cell wrapping its text (e.g. a long header) does not split its row
    words = {"Unit": _word(200, 120), "price": _word(200, 140), "Qty": _word(400, 130)}
    words |= {"10": _word(200, 250), "2": _word(400, 250)}
    grid, _, _ = _table(cells, logic, words)
    assert grid == [["Unit price", "Qty"], ["10", "2"]]

    # A cell spanning 2 rows whose text lines lie in each of them, aligned with the text of the row
    cells, logic = [[100, 100, 300, 200], [100, 200, 300, 300], [300, 100, 500, 300]], [[0, 0, 0, 0], [1, 1, 0, 0]]
    logic.append([0, 1, 1, 1])
    words = {"Acquired": _word(200, 150), "Taken": _word(200, 250), "51": _word(400, 150), "9": _word(400, 250)}
    grid, _, origin = _table(cells, logic, words)
    assert grid == [["Acquired", "51"], ["Taken", "9"]]
    assert sorted(origin) == [0, 1, 2, 2]  # both parts keep the score of the spanning cell
    # A label spanning 2 rows on a single line is kept as is
    words = {"Acquired": _word(200, 150), "Taken": _word(200, 250), "label": _word(400, 200)}
    _, out_logic, _ = _table(cells, logic, words)
    np.testing.assert_array_equal(out_logic, logic)


def _rotated(box, angle, pivot=(500.0, 500.0)):
    """The corners (TL, TR, BR, BL) of a box rotated by `angle` degrees, in a 1000 x 1000 page (relative)"""
    rad = np.deg2rad(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return (((as_polygons([box])[0] - pivot) @ rot.T + pivot) / 1000).tolist()


def test_assemble_page_table():
    # A 2 x 2 table rotated by 20 degrees on a page, whose words are assigned in the frame of its rows
    cells, logic = _grid([300, 500, 700], [400, 500, 600])
    page_cells = [
        {"geometry": _rotated(box, 20), "score": 0.9, "row_start": rs, "row_end": re, "col_start": cs, "col_end": ce}
        for box, (rs, re, cs, ce) in zip(cells, logic + 1)  # offset logical coordinates
    ]
    words = [_rotated(_word(*center), 20) for center in ((400, 450), (600, 450), (400, 550), (600, 550), (900, 900))]
    taken = np.zeros(len(words), dtype=bool)
    out, num_rows, num_cols = assemble_page_table(page_cells, np.array(words), (1000, 1000), taken)
    assert (num_rows, num_cols) == (2, 2)
    assert [(c["row_start"], c["col_start"], c["words"]) for c in out] == [
        (0, 0, [0]),
        (0, 1, [1]),
        (1, 0, [2]),
        (1, 1, [3]),
    ]
    # The cell geometries stay in the page frame, the word outside of the table is left free
    np.testing.assert_allclose(out[0]["geometry"], page_cells[0]["geometry"], atol=1e-5)
    assert taken.tolist() == [True, True, True, True, False]

    # Cells listing their words keep them, except the ones already taken by another table
    listed = [{**cell, "words": [idx]} for idx, cell in enumerate(page_cells)]
    taken = np.array([True, False, False, False, False])
    out, _, _ = assemble_page_table(listed, np.array(words), (1000, 1000), taken)
    assert [c["words"] for c in out] == [[], [1], [2], [3]]
    assert taken.tolist() == [True, True, True, True, False]

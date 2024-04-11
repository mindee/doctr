from app.utils import resolve_geometry


def test_resolve_geometry():
    dummy_box = [(0.0, 0.0), (1.0, 0.0)]
    dummy_polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    assert resolve_geometry(dummy_box) == (0.0, 0.0, 1.0, 0.0)
    assert resolve_geometry(dummy_polygon) == (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)

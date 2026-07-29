import cv2
import numpy as np
import pytest

from doctr.io import elements
from doctr.io.figures import (
    FigureEncoder,
    crop_layout_region,
    encode_crop,
    is_picture_label,
    is_picture_region,
    picture_regions,
)


def _page_image():
    """A dark page with a bright rectangle where the figure sits (relative box (0.1, 0.2) - (0.5, 0.6))"""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[20:60, 20:100] = (255, 0, 0)
    return image


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Picture", True),
        ("picture", True),
        ("Figure", True),
        ("Chart", True),
        ("Table", False),
        ("Text", False),
        ("Caption", False),
        (None, False),
    ],
)
def test_is_picture_label(label, expected):
    assert is_picture_label(label) is expected
    assert is_picture_region(elements.LayoutElement(label or "Text", 0.9, ((0, 0), (1, 1)))) is expected


def test_picture_regions():
    layout = [
        elements.LayoutElement("Text", 0.9, ((0.0, 0.0), (1.0, 0.1))),
        elements.LayoutElement("Picture", 0.9, ((0.1, 0.2), (0.5, 0.6))),
        elements.LayoutElement("Table", 0.9, ((0.1, 0.7), (0.9, 0.9))),
    ]
    page = elements.Page(_page_image(), [], 0, (100, 200), layout=layout)
    assert [region.type for region in picture_regions(page)] == ["Picture"]
    # A page without layout has no figure
    assert picture_regions(elements.Page(_page_image(), [], 0, (100, 200))) == []


def test_crop_layout_region():
    image = _page_image()
    crop = crop_layout_region(image, ((0.1, 0.2), (0.5, 0.6)))
    assert crop.shape[:2] == (41, 81)  # the crop bounds are inclusive
    assert (crop[:, :, 0] == 255).mean() > 0.9
    # Padding grows the region, so the dark background creeps in
    padded = crop_layout_region(image, ((0.1, 0.2), (0.5, 0.6)), padding=0.25)
    assert padded.shape[0] > crop.shape[0] and padded.shape[1] > crop.shape[1]
    assert (padded[:, :, 0] == 255).mean() < (crop[:, :, 0] == 255).mean()
    # Rotated polygons are de-rotated by a warp
    polygon = np.array([[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]], dtype=np.float32)
    assert crop_layout_region(image, polygon).shape[:2] == (40, 80)
    # A page without pixels, or a degenerate region, yields nothing
    assert crop_layout_region(None, ((0.1, 0.2), (0.5, 0.6))) is None
    assert crop_layout_region(np.zeros((0, 0, 3), dtype=np.uint8), ((0.1, 0.2), (0.5, 0.6))) is None
    assert crop_layout_region(image, ((0.5, 0.5), (0.5, 0.5))) is None


@pytest.mark.parametrize("image_format", ["png", "jpg", "jpeg", "webp"])
def test_encode_crop(image_format):
    crop = _page_image()[20:60, 20:100]
    payload = encode_crop(crop, image_format=image_format, quality=95)
    assert isinstance(payload, bytes) and len(payload) > 0
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == crop.shape
    # docTR pages are RGB: the red rectangle must survive the RGB -> BGR -> file -> BGR round trip
    assert decoded[..., 2].mean() > decoded[..., 0].mean()

    with pytest.raises(ValueError):
        encode_crop(crop, image_format="gif")


def test_figure_encoder_validation(tmp_path):
    with pytest.raises(ValueError):
        FigureEncoder(mode="inline")
    with pytest.raises(ValueError):
        FigureEncoder(mode="embedded", image_format="gif")
    with pytest.raises(ValueError):  # 'referenced' needs somewhere to write
        FigureEncoder(mode="referenced")
    FigureEncoder(mode="referenced", image_dir=tmp_path)

    # `resolve` accepts a mode, an encoder, or None
    assert FigureEncoder.resolve("embedded").mode == "embedded"
    assert FigureEncoder.resolve(None).mode == "none"
    encoder = FigureEncoder("placeholder")
    assert FigureEncoder.resolve(encoder) is encoder
    assert "placeholder" in repr(encoder)


def test_figure_encoder_modes(tmp_path):
    region = elements.LayoutElement("Picture", 0.9, ((0.1, 0.2), (0.5, 0.6)))
    page = elements.Page(_page_image(), [], 0, (100, 200), layout=[region])

    assert FigureEncoder("none").source(page, region, 1) is None
    assert not FigureEncoder("none").enabled
    assert FigureEncoder("placeholder").source(page, region, 1) is None
    assert FigureEncoder("placeholder").enabled and not FigureEncoder("placeholder").materializes

    embedded = FigureEncoder("embedded").source(page, region, 1)
    assert embedded.startswith("data:image/png;base64,")
    assert FigureEncoder("embedded", image_format="jpg").source(page, region, 1).startswith("data:image/jpeg;base64,")

    encoder = FigureEncoder("referenced", image_dir=tmp_path / "assets", path_prefix="assets/")
    assert encoder.source(page, region, 3) == "assets/page1_figure3.png"
    assert encoder.written == [tmp_path / "assets" / "page1_figure3.png"]
    assert encoder.written[0].read_bytes()[:4] == b"\x89PNG"

    # A page restored from a JSON export carries no pixels: the figures degrade to a placeholder
    restored = elements.Page.from_dict(page.export())
    assert FigureEncoder("embedded").source(restored, region, 1) is None
    assert FigureEncoder("embedded").materializes
    assert not FigureEncoder("embedded").materializes_on(restored)
    assert FigureEncoder("embedded").materializes_on(page)

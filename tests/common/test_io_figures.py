import gc
import re
import weakref
from pathlib import Path, PurePosixPath, PureWindowsPath

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
    # A sub-figure and a duplicate detection inside a picture are left out, a picture beside it is kept
    layout += [
        elements.LayoutElement("Picture", 0.8, ((0.15, 0.25), (0.3, 0.4))),
        elements.LayoutElement("Picture", 0.7, ((0.1, 0.2), (0.5, 0.6))),
        elements.LayoutElement("Picture", 0.6, ((0.55, 0.2), (0.9, 0.6))),
    ]
    page = elements.Page(_page_image(), [], 0, (100, 200), layout=layout)
    assert [region.confidence for region in picture_regions(page)] == [0.9, 0.6]
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
    assert crop_layout_region(image, np.zeros((3, 2), dtype=np.float32)) is None


@pytest.mark.parametrize("image_format", ["png", "jpg", "jpeg", "webp"])
def test_encode_crop(image_format):
    crop = _page_image()[20:60, 20:100]
    payload = encode_crop(crop, image_format=image_format, quality=95)
    assert isinstance(payload, bytes) and len(payload) > 0
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == crop.shape
    # docTR pages are RGB: the red rectangle must survive the RGB -> BGR -> file -> BGR round trip
    assert decoded[..., 2].mean() > decoded[..., 0].mean()
    rgba = np.dstack([crop, np.full(crop.shape[:2], 255, dtype=np.uint8)])
    decoded = cv2.imdecode(np.frombuffer(encode_crop(rgba, image_format), dtype=np.uint8), cv2.IMREAD_COLOR)
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
    # A negative padding would flip the region, and the encoders only accept a quality between 0 and 100
    for kwargs in ({"padding": -0.1}, {"quality": 101}, {"quality": -1}):
        with pytest.raises(ValueError):
            FigureEncoder(mode="embedded", **kwargs)
    with pytest.raises(ValueError):
        crop_layout_region(_page_image(), ((0.1, 0.2), (0.5, 0.6)), padding=-0.6)
    with pytest.raises(ValueError):
        encode_crop(_page_image(), "jpg", quality=500)

    # `resolve` accepts a mode, an encoder, or None
    assert FigureEncoder.resolve("embedded").mode == "embedded"
    assert FigureEncoder.resolve(None).mode == "none"
    # ... but not the 'referenced' mode, which needs an image directory: the error says how to configure it
    with pytest.raises(ValueError, match=r"FigureEncoder\('referenced', image_dir="):
        FigureEncoder.resolve("referenced")
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

    encoder = FigureEncoder("referenced", image_dir=tmp_path / "assets", path_prefix="my assets/")
    source = encoder.source(page, region, 3)
    # The file is named after the figure position and a hash of its content
    assert re.fullmatch(r"my%20assets/page1_figure3-[0-9a-f]{8}\.png", source)
    name = source.rsplit("/", 1)[-1]
    assert encoder.written == [tmp_path / "assets" / name]
    assert encoder.written[0].read_bytes()[:4] == b"\x89PNG"
    # Each figure is written once
    assert encoder.source(page, region, 3) == source
    assert len(encoder.written) == 1
    # Another page at the same position gets its own file, unless it holds the very same pixels
    other_img = _page_image()
    other_img[..., 1] = 255
    other = elements.Page(other_img, [], 0, (100, 200), layout=[region])
    other_source = encoder.source(other, region, 3)
    assert other_source != source and len(encoder.written) == 2
    twin = elements.Page(_page_image(), [], 0, (100, 200), layout=[region])
    assert encoder.source(twin, region, 3) == source

    # A page restored from a JSON export carries no pixels: the figures degrade to a placeholder
    restored = elements.Page.from_dict(page.export())
    assert FigureEncoder("embedded").source(restored, region, 1) is None
    assert FigureEncoder("embedded").materializes
    assert not FigureEncoder("embedded").materializes_on(restored)
    assert FigureEncoder("embedded").materializes_on(page)


def test_figure_encoder_does_not_pin_pages():
    # An encoder reused across a corpus must not keep every exported page (and its pixels) alive
    encoder = FigureEncoder("embedded")
    region = elements.LayoutElement("Picture", 0.9, ((0.1, 0.2), (0.5, 0.6)))
    page = elements.Page(_page_image(), [], 0, (100, 200), layout=[region])
    assert encoder.source(page, region, 1) is not None
    page_ref = weakref.ref(page)
    del page
    gc.collect()
    assert page_ref() is None
    assert encoder._sources == {}  # the cached source went away with its page
    # A new page (possibly recycling the id of the dead one) is encoded afresh
    other_img = _page_image()
    other_img[..., 2] = 255
    other = elements.Page(other_img, [], 0, (100, 200), layout=[region])
    assert encoder.source(other, region, 1) == FigureEncoder("embedded").source(other, region, 1)


def test_referenced_figures_of_several_documents_share_a_directory(tmp_path):
    # Two documents exported with their own encoder into the same directory must not overwrite each other
    region = elements.LayoutElement("Picture", 0.9, ((0.1, 0.2), (0.5, 0.6)))
    red, blue = np.zeros((100, 200, 3), dtype=np.uint8), np.zeros((100, 200, 3), dtype=np.uint8)
    red[..., 0], blue[..., 2] = 255, 255
    sources = []
    for image in (red, blue):
        page = elements.Page(image, [], 0, (100, 200), layout=[region])
        sources.append(FigureEncoder("referenced", image_dir=tmp_path).source(page, region, 1))
    assert sources[0] != sources[1]
    assert len(list(tmp_path.iterdir())) == 2
    # Each export still points at its own pixels (OpenCV reads BGR)
    assert cv2.imread(str(tmp_path / sources[0]))[..., 2].mean() > 200
    assert cv2.imread(str(tmp_path / sources[1]))[..., 0].mean() > 200
    # Re-exporting rewrites the very same file
    page = elements.Page(red, [], 0, (100, 200), layout=[region])
    assert FigureEncoder("referenced", image_dir=tmp_path).source(page, region, 1) == sources[0]
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize(
    "path_prefix, expected",
    [
        ("", ""),
        ("assets/", "assets/"),
        ("assets", "assets/"),  # a directory, the separator is added
        ("assets\\", "assets/"),  # Windows separators
        ("out\\my assets", "out/my%20assets/"),
        ("./assets", "./assets/"),
        ("../shared/figs/", "../shared/figs/"),
        (Path("assets"), "assets/"),
        (Path("out") / "assets", "out/assets/"),
        (PurePosixPath("out/assets"), "out/assets/"),
        (PureWindowsPath("out\\assets"), "out/assets/"),  # str(Path(...)) on Windows
        ("https://cdn.example.com/figs", "https://cdn.example.com/figs/"),
    ],
)
def test_figure_encoder_path_prefix_is_portable(tmp_path, path_prefix, expected):
    region = elements.LayoutElement("Picture", 0.9, ((0.1, 0.2), (0.5, 0.6)))
    page = elements.Page(_page_image(), [], 0, (100, 200), layout=[region])
    encoder = FigureEncoder("referenced", image_dir=tmp_path / "assets", path_prefix=path_prefix)
    source = encoder.source(page, region, 1)
    assert "\\" not in source and "%5C" not in source
    assert re.fullmatch(re.escape(expected) + r"page1_figure1-[0-9a-f]{8}\.png", source), source
    # The link resolves to the written file (relative prefixes)
    if expected.startswith("assets/"):
        from urllib.parse import unquote

        assert (tmp_path / unquote(source)).is_file()

from pathlib import Path

import numpy as np
import pytest

from doctr import datasets
from doctr.datasets.generator import base
from doctr.datasets.generator.base import _fonts_per_char, _renders_char, synthesize_text_img
from doctr.utils.fonts import get_font_candidates


@pytest.mark.parametrize("text", ["", "  ", "   "])
def test_synthesize_text_img_rejects_inkless_text(text):
    # Empty text, and text a font draws with no ink, both used to yield a 0-dimension image
    # and an opaque resize failure deep in a DataLoader worker (#2016).
    with pytest.raises(ValueError):
        synthesize_text_img(text)


def test_renders_char_smoke_on_a_real_font():
    # One real-font assertion, kept deliberately minimal: basic Latin coverage in the
    # default system font. It proves the Pillow/FreeType path works; every capability
    # rule below is asserted deterministically instead of through installed fonts.
    if not get_font_candidates():
        pytest.skip("no recommended system font installed")
    assert _renders_char("a", get_font_candidates()[0], 32)


def test_renders_char_rejects_blank_and_notdef(monkeypatch):
    # A character the font does not cover is drawn as a ".notdef" box: it carries ink, so
    # ink alone cannot tell it apart from a real glyph — it must be compared to the box.
    notdef = ((4, 4), b"notdef")
    ink = {"a": ((3, 5), b"glyph"), "?": notdef, " ": None}
    monkeypatch.setattr(base, "_glyph_ink", lambda font, char: ink[char])
    monkeypatch.setattr(base, "_notdef_ink", lambda family, size: frozenset({notdef}))
    base._renders_char.cache_clear()

    assert base._renders_char("a", None, 32)
    assert not base._renders_char("?", None, 32)  # .notdef box
    assert not base._renders_char(" ", None, 32)  # no ink at all
    base._renders_char.cache_clear()


def test_fonts_per_char_falls_back_to_system_fonts(monkeypatch):
    # "x" is drawn by neither given font, so it must fall back — and only to the system
    # candidate that actually draws it, in candidate order.
    monkeypatch.setattr(base, "get_font_candidates", lambda: ("blind.ttf", "rescue.ttf"))
    monkeypatch.setattr(base, "_renders_char", lambda char, font, size: char != "x" or font == "rescue.ttf")

    mapping = base._fonts_per_char("ax", ["given.ttf"])

    assert mapping["a"] == ("given.ttf",)
    assert mapping["x"] == ("rescue.ttf",)


def test_fonts_per_char_accepts_whitespace():
    # A space is inkless in every font by design; judging it by ink would reject any
    # vocabulary containing one, such as VOCABS["latex"].
    assert _fonts_per_char("a b", [None])[" "] == (None,)


def test_fonts_per_char_raises_when_no_font_renders(monkeypatch):
    monkeypatch.setattr(base, "get_font_candidates", lambda: ("blind.ttf",))
    monkeypatch.setattr(base, "_renders_char", lambda char, font, size: False)

    with pytest.raises(ValueError, match="cannot be rendered"):
        base._fonts_per_char("x", ["given.ttf"])


def test_visiondataset():
    url = "https://github.com/mindee/doctr/releases/download/v0.6.0/mnist.zip"
    with pytest.raises(ValueError):
        datasets.datasets.VisionDataset(url, download=False)

    dataset = datasets.datasets.VisionDataset(url, download=True, extract_archive=True)
    assert len(dataset) == 0
    assert repr(dataset) == "VisionDataset()"


def test_abstractdataset(mock_image_path):
    with pytest.raises(ValueError):
        datasets.datasets.AbstractDataset("my/fantasy/folder")

    # Check transforms
    path = Path(mock_image_path)
    ds = datasets.datasets.AbstractDataset(path.parent)
    # Check target format
    with pytest.raises(AssertionError):
        ds.data = [(path.name, 0)]
        _ = ds[0]
    with pytest.raises(AssertionError):
        ds.data = [(path.name, dict(boxes=np.array([[0, 0, 1, 1]])))]
        _ = ds[0]
    with pytest.raises(AssertionError):
        ds.data = [(path.name, {"label": "A"})]
        _ = ds[0]

    # Patch some data
    ds.data = [(path.name, np.array([0]))]

    # Fetch the img
    sample = ds[0]
    img, target = sample.image, sample.target
    assert isinstance(target, np.ndarray)
    assert np.array_equal(target, np.array([0]))

    # Check img_transforms
    def img_transform(sample):
        sample.image = 1 - sample.image
        return sample

    ds.img_transforms = img_transform

    sample2 = ds[0]
    img2, target2 = sample2.image, sample2.target

    assert np.all(img2.numpy() == 1 - img.numpy())
    assert np.array_equal(target, target2)

    # Check sample_transforms
    ds.img_transforms = None

    def sample_transform(sample):
        sample.target = sample.target + 1
        return sample

    ds.sample_transforms = sample_transform

    sample3 = ds[0]
    img3, target3 = sample3.image, sample3.target

    assert np.all(img3.numpy() == img.numpy())
    assert np.array_equal(target3, target + 1)

    # Check inplace modifications
    ds.data = [(ds.data[0][0], "A")]

    def inplace_transfo(sample):
        sample.target += "B"
        return sample

    ds.sample_transforms = inplace_transfo

    t = ds[0].target
    t = ds[0].target

    assert t == "AB"

import pytest
from PIL.ImageFont import FreeTypeFont, ImageFont

from doctr.utils import fonts
from doctr.utils.fonts import get_font


@pytest.fixture(autouse=True)
def _reset_font_cache():
    # Ensure each test starts with a fresh font resolution cache
    fonts._resolve_default_font_family.cache_clear()
    yield
    fonts._resolve_default_font_family.cache_clear()


def test_get_font_default():
    # Attempts to load recommended OS font
    font = get_font()

    assert isinstance(font, (ImageFont, FreeTypeFont))
    # The font must be able to measure text
    x0, y0, x1, y1 = font.getbbox("hello")
    assert x1 > x0 and y1 > y0


def test_get_font_respects_size():
    font = get_font(font_size=32)
    # Both system fonts and Pillow >= 10.1 scalable default expose `size`
    if hasattr(font, "size"):
        assert font.size == 32


def test_get_font_explicit_family():
    # An explicitly requested font that exists should load
    default_family = fonts._resolve_default_font_family()
    if default_family is not None:
        font = get_font(default_family, 16)
        assert isinstance(font, FreeTypeFont)
        assert font.size == 16

    # An explicitly requested font that does not exist should fail loudly
    with pytest.raises(OSError):
        get_font("this-font-does-not-exist.ttf")


def test_get_font_resolution_is_cached():
    get_font()
    info_after_first = fonts._resolve_default_font_family.cache_info()
    get_font()
    get_font(font_size=24)
    info_after_more = fonts._resolve_default_font_family.cache_info()

    # The filesystem probing must run at most once per process
    assert info_after_first.misses == 1
    assert info_after_more.misses == 1
    assert info_after_more.hits >= info_after_first.hits + 2


def test_get_font_fallback(monkeypatch):
    # Force every candidate to be unavailable so the built-in fallback is exercised
    monkeypatch.setattr(fonts, "_FONT_CANDIDATES", dict.fromkeys(fonts._FONT_CANDIDATES, ("missing-font.ttf",)))
    fonts._resolve_default_font_family.cache_clear()

    font = get_font(font_size=20)
    assert isinstance(font, (ImageFont, FreeTypeFont))
    # The fallback font must still be usable for text measurement
    assert font.getbbox("hello")[2] > 0

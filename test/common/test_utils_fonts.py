from PIL.ImageFont import ImageFont, FreeTypeFont

from doctr.utils.fonts import get_font


def test_get_font():

    # Attempts to load recommended OS font
    font = get_font()

    assert isinstance(font, (ImageFont, FreeTypeFont))

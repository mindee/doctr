# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
import platform
from typing import Optional

from PIL import ImageFont

__all__ = ['get_font']


def get_font(font_family: Optional[str] = None, font_size: int = 13) -> ImageFont.ImageFont:
    """Resolves a compatible ImageFont for the system

    Args:
        font_family: the font family to use
        font_size: the size of the font upon rendering

    Returns:
        the Pillow font
    """

    # Font selection
    if font_family is None:
        try:
            font = ImageFont.truetype("FreeMono.ttf" if platform.system() == "Linux" else "Arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
            logging.warning("unable to load recommended font family. Loading default PIL font,"
                            "font size issues may be expected."
                            "To prevent this, it is recommended to specify the value of 'font_family'.")
    else:
        font = ImageFont.truetype(font_family, font_size)

    return font

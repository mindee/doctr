# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import platform
import logging
from PIL import ImageFont
from typing import Optional

__all__ = ['get_font']


def get_font(font_family: Optional[str] = None, font_size: int = 13) -> ImageFont.ImageFont:

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

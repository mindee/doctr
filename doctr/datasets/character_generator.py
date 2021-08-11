# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from PIL import ImageFont, ImageDraw, Image
import requests
import io
from typing import Tuple


__all__ = ['generate_character']


def generate_character(
    char: str = None,
    size: int = 32,
    font_url: str = 'https://github.com/ProgrammingFonts/ProgrammingFonts/raw/master/'
                    'Droid-Sans-Mono/droid-sans-mono-1.00/Droid%20Sans%20Mono.ttf',
    inverted: bool = False,
    angle: int = 0,
) -> Tuple[np.ndarray, str]:

    """Generate a MNIST-like square picture of 1 randomly chosen character in the vocab.

    Args:
        char: string of the char to draw
        size: size of the square image to draw
        font_url: truetype font (.ttf) to use
        inverted: bool, if True image background is black and char filled in white.
        angle: if greater than 0, rotate the image of the angle (degrees)

    Returns:
        A tuple: PIL Image, character
    """
    # Template
    back_color = (0, 0, 0) if inverted else (255, 255, 255)
    img = Image.new('RGB', (size, size), color=back_color)
    d = ImageDraw.Draw(img)

    # Load font from URL
    r = requests.get(font_url, allow_redirects=True)
    font = ImageFont.truetype(io.BytesIO(r.content), size=size - 4)

    # Draw the character
    fill_color = (255, 255, 255) if inverted else (0, 0, 0)
    d.text((4, 0), char, font=font, fill=fill_color)

    # Rotate
    if angle > 0:
        img = img.rotate(angle, resample=Image.NEAREST, expand=False)

    # Return image & annotation
    return np.asarray(img), char

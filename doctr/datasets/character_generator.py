# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import PIL
from PIL import ImageFont, ImageDraw
from PIL.Image import Image
import random
import requests
import io
from vocabs import VOCABS
from typing import Tuple


def generate_character(
    vocab: str = VOCABS["french"][:-1],
    size: int = 32,
    font_url: str = 'https://github.com/ProgrammingFonts/ProgrammingFonts/raw/master/'
                    'Droid-Sans-Mono/droid-sans-mono-1.00/Droid%20Sans%20Mono.ttf',
    inverted: bool = False,
    max_angle: int = 0,
) -> Tuple[Image, str]:

    """Generate a MNIST-like square picture of 1 randomly chosen character in the vocab.

    Args:
        vocab: string of the vocabulary
        size: size of the square image to draw
        font_url: truetype font (.ttf) to use
        inverted: bool, if True image background is black and char filled in white.
        max_angle: if greater than 0, randomly rotate the image of an angle between [-max_angle, max_angle] (degrees)

    Returns:
        A tuple: PIL Image, character
    """
    # Template
    back_color = (0, 0, 0) if inverted else (255, 255, 255)
    img = PIL.Image.new('RGB', (size, size), color=back_color)
    d = ImageDraw.Draw(img)

    # Load font from URL
    r = requests.get(font_url, allow_redirects=True)
    font = ImageFont.truetype(io.BytesIO(r.content), size=size - 4)

    # Draw the character
    char = random.choice(vocab)
    fill_color = (255, 255, 255) if inverted else (0, 0, 0)
    d.text((4, 0), char, font=font, fill=fill_color)

    # Rotate
    if max_angle > 0:
        angle = random.randrange(-max_angle, max_angle, 1)
        img = img.rotate(angle, resample=PIL.Image.NEAREST, expand=False)

    # Return image & annotation
    return img, char

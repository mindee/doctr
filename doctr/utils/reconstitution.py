# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
from typing import Any, Dict, Optional

import numpy as np
from anyascii import anyascii
from PIL import Image, ImageDraw

from .fonts import get_font

__all__ = ["synthesize_page", "synthesize_kie_page"]


def synthesize_page(
    page: Dict[str, Any],
    draw_proba: bool = False,
    font_family: Optional[str] = None,
) -> np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
    ----
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_size: size of the font, default font = 13
        font_family: family of the font

    Returns:
    -------
        the synthesized page
    """
    # Draw template
    h, w = page["dimensions"]
    response = 255 * np.ones((h, w, 3), dtype=np.int32)

    # Draw each word
    for block in page["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                # Get absolute word geometry
                (xmin, ymin), (xmax, ymax) = word["geometry"]
                xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
                ymin, ymax = int(round(h * ymin)), int(round(h * ymax))

                # White drawing context adapted to font size, 0.75 factor to convert pts --> pix
                font = get_font(font_family, int(0.75 * (ymax - ymin)))
                img = Image.new("RGB", (xmax - xmin, ymax - ymin), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                # Draw in black the value of the word
                try:
                    d.text((0, 0), word["value"], font=font, fill=(0, 0, 0))
                except UnicodeEncodeError:
                    # When character cannot be encoded, use its anyascii version
                    d.text((0, 0), anyascii(word["value"]), font=font, fill=(0, 0, 0))

                # Colorize if draw_proba
                if draw_proba:
                    p = int(255 * word["confidence"])
                    mask = np.where(np.array(img) == 0, 1, 0)
                    proba: np.ndarray = np.array([255 - p, 0, p])
                    color = mask * proba[np.newaxis, np.newaxis, :]
                    white_mask = 255 * (1 - mask)
                    img = color + white_mask

                # Write to response page
                response[ymin:ymax, xmin:xmax, :] = np.array(img)

    return response


def synthesize_kie_page(
    page: Dict[str, Any],
    draw_proba: bool = False,
    font_family: Optional[str] = None,
) -> np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
    ----
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_size: size of the font, default font = 13
        font_family: family of the font

    Returns:
    -------
        the synthesized page
    """
    # Draw template
    h, w = page["dimensions"]
    response = 255 * np.ones((h, w, 3), dtype=np.int32)

    # Draw each word
    for predictions in page["predictions"].values():
        for prediction in predictions:
            # Get aboslute word geometry
            (xmin, ymin), (xmax, ymax) = prediction["geometry"]
            xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
            ymin, ymax = int(round(h * ymin)), int(round(h * ymax))

            # White drawing context adapted to font size, 0.75 factor to convert pts --> pix
            font = get_font(font_family, int(0.75 * (ymax - ymin)))
            img = Image.new("RGB", (xmax - xmin, ymax - ymin), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            # Draw in black the value of the word
            try:
                d.text((0, 0), prediction["value"], font=font, fill=(0, 0, 0))
            except UnicodeEncodeError:
                # When character cannot be encoded, use its anyascii version
                d.text((0, 0), anyascii(prediction["value"]), font=font, fill=(0, 0, 0))

            # Colorize if draw_proba
            if draw_proba:
                p = int(255 * prediction["confidence"])
                mask = np.where(np.array(img) == 0, 1, 0)
                proba: np.ndarray = np.array([255 - p, 0, p])
                color = mask * proba[np.newaxis, np.newaxis, :]
                white_mask = 255 * (1 - mask)
                img = color + white_mask

            # Write to response page
            response[ymin:ymax, xmin:xmax, :] = np.array(img)

    return response

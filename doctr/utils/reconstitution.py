# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
import logging
from typing import Any

import numpy as np
from anyascii import anyascii
from PIL import Image, ImageDraw

from .fonts import get_font

__all__ = ["synthesize_page", "synthesize_kie_page"]


# Global variable to avoid multiple warnings
ROTATION_WARNING = False


def _warn_rotation(entry: dict[str, Any]) -> None:  # pragma: no cover
    global ROTATION_WARNING
    if not ROTATION_WARNING and len(entry["geometry"]) == 4:
        logging.warning("Polygons with larger rotations will lead to inaccurate rendering")
        ROTATION_WARNING = True


def _synthesize(
    response: Image.Image,
    entry: dict[str, Any],
    w: int,
    h: int,
    draw_proba: bool = False,
    font_family: str | None = None,
    smoothing_factor: float = 0.75,
    min_font_size: int = 6,
    max_font_size: int = 50,
) -> Image.Image:
    if len(entry["geometry"]) == 2:
        (xmin, ymin), (xmax, ymax) = entry["geometry"]
        polygon = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    else:
        polygon = entry["geometry"]

    # Calculate the bounding box of the word
    x_coords, y_coords = zip(*polygon)
    xmin, ymin, xmax, ymax = (
        int(round(w * min(x_coords))),
        int(round(h * min(y_coords))),
        int(round(w * max(x_coords))),
        int(round(h * max(y_coords))),
    )
    word_width = xmax - xmin
    word_height = ymax - ymin

    # If lines are provided instead of words, concatenate the word entries
    if "words" in entry:
        word_text = " ".join(word["value"] for word in entry["words"])
    else:
        word_text = entry["value"]
    # Find the optimal font size
    try:
        font_size = min(word_height, max_font_size)
        font = get_font(font_family, font_size)
        text_width, text_height = font.getbbox(word_text)[2:4]

        while (text_width > word_width or text_height > word_height) and font_size > min_font_size:
            font_size = max(int(font_size * smoothing_factor), min_font_size)
            font = get_font(font_family, font_size)
            text_width, text_height = font.getbbox(word_text)[2:4]
    except ValueError:
        font = get_font(font_family, min_font_size)

    # Create a mask for the word
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon([(int(round(w * x)), int(round(h * y))) for x, y in polygon], fill=255)

    # Draw the word text
    d = ImageDraw.Draw(response)
    try:
        try:
            d.text((xmin, ymin), word_text, font=font, fill=(0, 0, 0), anchor="lt")
        except UnicodeEncodeError:
            d.text((xmin, ymin), anyascii(word_text), font=font, fill=(0, 0, 0), anchor="lt")
    # Catch generic exceptions to avoid crashing the whole rendering
    except Exception:  # pragma: no cover
        logging.warning(f"Could not render word: {word_text}")

    if draw_proba:
        confidence = (
            entry["confidence"]
            if "confidence" in entry
            else sum(w["confidence"] for w in entry["words"]) / len(entry["words"])
        )
        p = int(255 * confidence)
        color = (255 - p, 0, p)  # Red to blue gradient based on probability
        d.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

        prob_font = get_font(font_family, 20)
        prob_text = f"{confidence:.2f}"
        prob_text_width, prob_text_height = prob_font.getbbox(prob_text)[2:4]

        # Position the probability slightly above the bounding box
        prob_x_offset = (word_width - prob_text_width) // 2
        prob_y_offset = ymin - prob_text_height - 2
        prob_y_offset = max(0, prob_y_offset)

        d.text((xmin + prob_x_offset, prob_y_offset), prob_text, font=prob_font, fill=color, anchor="lt")

    return response


def synthesize_page(
    page: dict[str, Any],
    draw_proba: bool = False,
    font_family: str | None = None,
    smoothing_factor: float = 0.95,
    min_font_size: int = 8,
    max_font_size: int = 50,
) -> np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_family: family of the font
        smoothing_factor: factor to smooth the font size
        min_font_size: minimum font size
        max_font_size: maximum font size

    Returns:
        the synthesized page
    """
    # Draw template
    h, w = page["dimensions"]
    response = Image.new("RGB", (w, h), color=(255, 255, 255))

    for block in page["blocks"]:
        # If lines are provided use these to get better rendering results
        if len(block["lines"]) > 1:
            for line in block["lines"]:
                _warn_rotation(block)  # pragma: no cover
                response = _synthesize(
                    response=response,
                    entry=line,
                    w=w,
                    h=h,
                    draw_proba=draw_proba,
                    font_family=font_family,
                    smoothing_factor=smoothing_factor,
                    min_font_size=min_font_size,
                    max_font_size=max_font_size,
                )
        # Otherwise, draw each word
        else:
            for line in block["lines"]:
                _warn_rotation(block)  # pragma: no cover
                for word in line["words"]:
                    response = _synthesize(
                        response=response,
                        entry=word,
                        w=w,
                        h=h,
                        draw_proba=draw_proba,
                        font_family=font_family,
                        smoothing_factor=smoothing_factor,
                        min_font_size=min_font_size,
                        max_font_size=max_font_size,
                    )

    return np.array(response, dtype=np.uint8)


def synthesize_kie_page(
    page: dict[str, Any],
    draw_proba: bool = False,
    font_family: str | None = None,
) -> np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_family: family of the font
        smoothing_factor: factor to smooth the font size
        min_font_size: minimum font size
        max_font_size: maximum font size

    Returns:
        the synthesized page
    """
    # Draw template
    h, w = page["dimensions"]
    response = Image.new("RGB", (w, h), color=(255, 255, 255))

    # Draw each word
    for predictions in page["predictions"].values():
        for prediction in predictions:
            _warn_rotation(prediction)  # pragma: no cover
            response = _synthesize(
                response=response,
                entry=prediction,
                w=w,
                h=h,
                draw_proba=draw_proba,
                font_family=font_family,
            )
    return np.array(response, dtype=np.uint8)

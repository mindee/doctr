# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
import logging
import math
from functools import lru_cache
from typing import Any

import numpy as np
from anyascii import anyascii
from PIL import Image, ImageDraw, ImageFont

from .fonts import get_font

__all__ = ["synthesize_page", "synthesize_kie_page"]


@lru_cache(maxsize=256)
def _cached_font(font_family: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Memoized font loader: avoids re-reading the font file for every word."""
    return get_font(font_family, font_size)


@lru_cache(maxsize=1)
def _warn_rotation_once() -> None:  # pragma: no cover
    # lru_cache gives us thread-safe "warn once" semantics without a mutable global
    logging.warning("Polygons with larger rotations may lead to slightly inaccurate rendering")


def _polygon_angle(polygon: list[tuple[float, float]], w: int, h: int) -> float:
    """Estimate the rotation angle (degrees, counter-clockwise) from the top edge of a 4-point polygon."""
    (x0, y0), (x1, y1) = polygon[0], polygon[1]
    return -math.degrees(math.atan2((y1 - y0) * h, (x1 - x0) * w))


def _text_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    bbox = font.getbbox(text)
    return max(int(bbox[2]) - int(bbox[0]), 1)


def _fit_font(
    text: str,
    box_w: int,
    box_h: int,
    font_family: str | None,
    min_font_size: int,
    max_font_size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Directly estimate the largest font size fitting the box (text width scales ~linearly with size)."""
    font_size = max(min(box_h, max_font_size), min_font_size)
    try:
        font = _cached_font(font_family, font_size)
        x0, y0, x1, y1 = font.getbbox(text)
        text_w, text_h = max(int(x1) - int(x0), 1), max(int(y1) - int(y0), 1)
        if text_w > box_w or text_h > box_h:
            scale = min(box_w / text_w, box_h / text_h)
            font_size = max(min(int(font_size * scale), max_font_size), min_font_size)
            font = _cached_font(font_family, font_size)
        # The linear estimate can be off by a pixel or two: shrink until the text truly fits
        while font_size > min_font_size and _text_width(font, text) > box_w:
            font_size -= 1
            font = _cached_font(font_family, font_size)
    except ValueError:
        font = _cached_font(font_family, min_font_size)
    return font


def _fit_line_font(
    word_boxes: list[tuple[str, int, int, int, int]],
    line_height: int,
    font_family: str | None,
    min_font_size: int,
    max_font_size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find one font size for a whole line such that every word fits its own bounding box."""
    font_size = max(min(line_height, max_font_size), min_font_size)
    try:
        font = _cached_font(font_family, font_size)
        # Scale down so the most constrained word still fits its own box
        scale = min([(xmax - xmin) / _text_width(font, value) for value, xmin, _, xmax, _ in word_boxes] + [1.0])
        if scale < 1.0:
            font_size = max(min(int(font_size * scale), max_font_size), min_font_size)
            font = _cached_font(font_family, font_size)
        # The linear estimate can be off by a pixel or two: shrink until every word truly fits
        while font_size > min_font_size and any(
            _text_width(font, value) > (xmax - xmin) for value, xmin, _, xmax, _ in word_boxes
        ):
            font_size -= 1
            font = _cached_font(font_family, font_size)
    except ValueError:
        font = _cached_font(font_family, min_font_size)
    return font


def _draw_word(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    anchor: str = "lm",
) -> None:
    try:
        try:
            d.text(xy, text, font=font, fill=fill, anchor=anchor)
        except UnicodeEncodeError:
            d.text(xy, anyascii(text), font=font, fill=fill, anchor=anchor)
    except Exception:  # pragma: no cover
        logging.warning(f"Could not render word: {text}")


def _paste_rotated_word(
    response: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    center: tuple[int, int],
    angle: float,
    fill: tuple[int, int, int],
) -> None:
    """Render a word on a transparent patch, rotate it, and paste it centered on the polygon centroid."""
    bbox = font.getbbox(text)
    x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    patch = Image.new("RGBA", (max(x1 - x0, 1) + 4, max(y1 - y0, 1) + 4), (0, 0, 0, 0))
    _draw_word(ImageDraw.Draw(patch), (2 - x0, 2 - y0), text, font, fill, anchor="la")
    patch = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    response.paste(patch, (center[0] - patch.width // 2, center[1] - patch.height // 2), patch)


def _synthesize(
    response: Image.Image,
    entry: dict[str, Any],
    w: int,
    h: int,
    draw_proba: bool = False,
    font_family: str | None = None,
    min_font_size: int = 6,
    max_font_size: int = 50,
    text_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if len(entry["geometry"]) == 2:
        (xmin_r, ymin_r), (xmax_r, ymax_r) = entry["geometry"]
        polygon = [(xmin_r, ymin_r), (xmax_r, ymin_r), (xmax_r, ymax_r), (xmin_r, ymax_r)]
        angle = 0.0
    else:
        polygon = entry["geometry"]
        angle = _polygon_angle(polygon, w, h)

    # Calculate the bounding box of the entry
    x_coords, y_coords = zip(*polygon)
    xmin, ymin, xmax, ymax = (
        int(round(w * min(x_coords))),
        int(round(h * min(y_coords))),
        int(round(w * max(x_coords))),
        int(round(h * max(y_coords))),
    )
    box_width, box_height = max(xmax - xmin, 1), max(ymax - ymin, 1)

    d = ImageDraw.Draw(response)

    if "words" in entry:
        # Line entry: one consistent font size for the whole line, drawn word by word.
        # The font is sized so that no word overflows its own box (no overlap between words).
        word_boxes: list[tuple[str, int, int, int, int]] = []
        for word in entry["words"]:
            wx, wy = zip(*word["geometry"])
            word_boxes.append((
                word["value"],
                int(round(w * min(wx))),
                int(round(h * min(wy))),
                int(round(w * max(wx))),
                int(round(h * max(wy))),
            ))
        font = _fit_line_font(word_boxes, box_height, font_family, min_font_size, max_font_size)
        for value, wxmin, wymin, _, wymax in word_boxes:
            _draw_word(d, (wxmin, (wymin + wymax) // 2), value, font, text_color, anchor="lm")
    else:
        word_text = entry["value"]
        if abs(angle) > 3:  # Rotated word: render on a patch and paste it rotated
            font = _fit_font(word_text, box_width, box_height, font_family, min_font_size, max_font_size)
            cx, cy = int(round(w * sum(x_coords) / len(x_coords))), int(round(h * sum(y_coords) / len(y_coords)))
            _paste_rotated_word(response, word_text, font, (cx, cy), angle, text_color)
        else:
            font = _fit_font(word_text, box_width, box_height, font_family, min_font_size, max_font_size)
            # "lm" anchor: vertically centered in the box, no ascender-offset drift
            _draw_word(d, (xmin, (ymin + ymax) // 2), word_text, font, text_color, anchor="lm")

    if draw_proba:
        confidence = (
            entry["confidence"]
            if "confidence" in entry
            else sum(word["confidence"] for word in entry["words"]) / len(entry["words"])
        )
        p = int(255 * confidence)
        color = (255 - p, 0, p)  # Red to blue gradient based on probability
        d.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

        # Scale the confidence label with the box instead of a hardcoded size
        prob_font = _cached_font(font_family, max(min(box_height // 2, 20), 10))
        prob_text = f"{confidence:.2f}"
        prob_text_width, prob_text_height = prob_font.getbbox(prob_text)[2:4]
        prob_x_offset = (box_width - prob_text_width) // 2
        prob_y_offset = max(0, ymin - prob_text_height - 2)
        d.text((xmin + prob_x_offset, prob_y_offset), prob_text, font=prob_font, fill=color, anchor="lt")

    return response


def synthesize_page(
    page: dict[str, Any],
    draw_proba: bool = False,
    font_family: str | None = None,
    min_font_size: int = 8,
    max_font_size: int = 50,
    background_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Draw the content of the element page (OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_family: family of the font
        min_font_size: minimum font size
        max_font_size: maximum font size
        background_color: RGB color of the page background
        text_color: RGB color of the rendered text

    Returns:
        the synthesized page
    """
    h, w = page["dimensions"]
    response = Image.new("RGB", (w, h), color=background_color)

    for block in page["blocks"]:
        for line in block["lines"]:
            if len(line["geometry"]) == 4:
                _warn_rotation_once()  # pragma: no cover
            # Line-level entry keeps a consistent font per line while preserving word positions
            response = _synthesize(
                response=response,
                entry=line,
                w=w,
                h=h,
                draw_proba=draw_proba,
                font_family=font_family,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                text_color=text_color,
            )

    return np.array(response, dtype=np.uint8)


def synthesize_kie_page(
    page: dict[str, Any],
    draw_proba: bool = False,
    font_family: str | None = None,
    min_font_size: int = 8,
    max_font_size: int = 50,
    background_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Draw the content of the element page (KIE OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_family: family of the font
        min_font_size: minimum font size
        max_font_size: maximum font size
        background_color: RGB color of the page background
        text_color: RGB color of the rendered text

    Returns:
        the synthesized page
    """
    h, w = page["dimensions"]
    response = Image.new("RGB", (w, h), color=background_color)

    for predictions in page["predictions"].values():
        for prediction in predictions:
            if len(prediction["geometry"]) == 4:
                _warn_rotation_once()  # pragma: no cover
            response = _synthesize(
                response=response,
                entry=prediction,
                w=w,
                h=h,
                draw_proba=draw_proba,
                font_family=font_family,
                min_font_size=min_font_size,
                max_font_size=max_font_size,
                text_color=text_color,
            )
    return np.array(response, dtype=np.uint8)

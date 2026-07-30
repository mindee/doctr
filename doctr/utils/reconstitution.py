# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
import logging
import math
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
from anyascii import anyascii
from PIL import Image, ImageDraw, ImageFont

from .fonts import get_font

__all__ = ["synthesize_page", "synthesize_kie_page"]


class _Word(NamedTuple):
    """A word to render: its text, where it starts, and how much room it was detected in."""

    value: str
    x: int  # anchor: middle of the leading edge of the box, like the "lm" anchor of flat text
    y: int
    width: int  # extent along the text direction
    height: int  # extent across the text direction
    angle: float  # degrees, counter-clockwise


@lru_cache(maxsize=256)
def _cached_font(font_family: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Memoized font loader: avoids re-reading the font file for every word."""
    try:
        return get_font(font_family, max(font_size, 1))
    except Exception:  # noqa: BLE001 - a missing or broken font must not abort a whole page
        logging.warning(f"Could not load font '{font_family}', falling back to the default font")
        return get_font(None, max(font_size, 1))


@lru_cache(maxsize=1)
def _warn_rotation_once() -> None:  # pragma: no cover
    # lru_cache gives us thread-safe "warn once" semantics without a mutable global
    logging.warning("Polygons with larger rotations may lead to slightly inaccurate rendering")


def _points(geometry: Any) -> list[tuple[float, float]] | None:
    try:
        points = [(float(x), float(y)) for x, y in geometry]
    except (TypeError, ValueError):
        return None
    if len(points) not in (2, 4) or not all(math.isfinite(v) for point in points for v in point):
        return None
    return points


def _polygon_angle(polygon: list[tuple[float, float]], w: int, h: int) -> float:
    (x0, y0), (x1, y1) = polygon[0], polygon[1]
    return -math.degrees(math.atan2((y1 - y0) * h, (x1 - x0) * w))


def _text_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    bbox = font.getbbox(text)
    return max(math.ceil(bbox[2]), 1)


def _text_height(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    bbox = font.getbbox(text)
    return max(int(bbox[3]) - int(bbox[1]), 1)


def _text_vspan(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    """Ascender-to-descender span: the vertical extent the "lm" anchor is centered on."""
    try:
        ascent, descent = font.getmetrics()
        return max(ascent + descent, 1)
    except AttributeError:  # pragma: no cover - bitmap fonts expose no metrics
        return _text_height(font, text)


def _fit_font_size(
    text: str,
    box_w: int,
    box_h: int,
    font_family: str | None,
    min_font_size: int,
    max_font_size: int,
) -> int:
    """Directly estimate the largest font size fitting the box (text width scales ~linearly with size)."""
    font_size = max(min(box_h, max_font_size), min_font_size)
    try:
        font = _cached_font(font_family, font_size)
        # A font size is an em size while the box bounds the ink, so scale on the measured ink:
        # without this the text is rendered noticeably smaller than the box it was detected in.
        scale = min(box_w / _text_width(font, text), box_h / _text_height(font, text))
        font_size = max(min(int(font_size * scale), max_font_size), min_font_size)
    except ValueError:  # pragma: no cover
        font_size = min_font_size
    return font_size


def _fit_line_font_size(
    words: list[_Word],
    font_family: str | None,
    min_font_size: int,
    max_font_size: int,
) -> int:
    """Find one font size for a whole line, driven by the median word box."""
    font_size = max(min(int(np.median([word.height for word in words])), max_font_size), min_font_size)
    try:
        font = _cached_font(font_family, font_size)
        scale = min(
            float(np.median([word.height / _text_height(font, word.value) for word in words])),
            float(np.median([word.width / _text_width(font, word.value) for word in words])),
        )
        font_size = max(min(int(font_size * scale), max_font_size), min_font_size)
    except ValueError:  # pragma: no cover
        font_size = min_font_size
    return font_size


def _harmonize(sizes: list[int], tolerance: float = 0.2) -> list[int]:
    """Snap font sizes that are close to each other onto one common size.

    Boxes for one and the same typeface come out a pixel or two apart from line to line, and
    rendering every line at its own size is what makes a synthesized page look ragged. Sizes
    within `tolerance` of each other collapse onto their median, while genuinely different sizes
    - a headline, a caption, small print - stay apart.
    """
    values = np.array(sizes, dtype=float)
    return [int(round(float(np.median(values[np.abs(values - size) <= tolerance * size])))) for size in sizes]


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
        try:
            # Anchors are rejected by bitmap fonts, which would otherwise leave the page blank
            d.text(xy, anyascii(text), font=font, fill=fill)
        except Exception:
            logging.warning(f"Could not render word: {text}")


def _paste_word(
    response: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    position: tuple[int, int],
    fill: tuple[int, int, int],
    angle: float = 0.0,
    squeeze: float = 1.0,
) -> None:
    """Render a word on a transparent patch, condense and rotate it, then paste it.

    `position` is the middle of the left edge of the text, the same anchor point as a flat "lm"
    draw. `squeeze` condenses the glyphs horizontally without touching their height, which is how
    a line that is too long for its box stays inside it at the size the rest of the page uses.
    """
    pad = 2
    patch = Image.new("RGBA", (_text_width(font, text) + 2 * pad, _text_vspan(font, text) + 2 * pad), (0, 0, 0, 0))
    _draw_word(ImageDraw.Draw(patch), (pad, pad), text, font, fill, anchor="la")
    if squeeze < 1.0:
        patch = patch.resize((max(round(patch.width * squeeze), 1), patch.height), Image.Resampling.BICUBIC)
        pad = round(pad * squeeze)

    anchor_x, anchor_y = float(pad), patch.height / 2
    if abs(angle) > 3:
        # rotate() turns the patch about its centre and expands the canvas, so follow the anchor
        theta = math.radians(angle)
        dx, dy = anchor_x - patch.width / 2, anchor_y - patch.height / 2
        patch = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        anchor_x = patch.width / 2 + dx * math.cos(theta) + dy * math.sin(theta)
        anchor_y = patch.height / 2 - dx * math.sin(theta) + dy * math.cos(theta)
    response.paste(patch, (round(position[0] - anchor_x), round(position[1] - anchor_y)), patch)


def _entry_words(entry: dict[str, Any], w: int, h: int) -> list[_Word]:
    """Collect the renderable words of a line entry, in pixel coordinates."""
    words = []
    for word in entry.get("words") or []:
        if not isinstance(word, dict):
            continue
        geom = _points(word.get("geometry"))
        if geom is None or not str(word.get("value", "")).strip():
            continue
        value = str(word["value"])
        if len(geom) == 2:
            (gx0, gy0), (gx1, gy1) = geom
            x0, y0 = int(round(w * gx0)), int(round(h * gy0))
            x1, y1 = int(round(w * gx1)), int(round(h * gy1))
            words.append(_Word(value, min(x0, x1), (y0 + y1) // 2, max(abs(x1 - x0), 1), max(abs(y1 - y0), 1), 0.0))
        else:
            # True text-direction extent: length of the top edge / left edge in pixels
            width = math.hypot((geom[1][0] - geom[0][0]) * w, (geom[1][1] - geom[0][1]) * h)
            height = math.hypot((geom[2][0] - geom[1][0]) * w, (geom[2][1] - geom[1][1]) * h)
            words.append(
                _Word(
                    value,
                    int(round(w * (geom[0][0] + geom[3][0]) / 2)),
                    int(round(h * (geom[0][1] + geom[3][1]) / 2)),
                    max(int(round(width)), 1),
                    max(int(round(height)), 1),
                    _polygon_angle(geom, w, h),
                )
            )
    return words


def _line_axis(words: list[_Word]) -> tuple[float, float, float, float, float]:
    """The baseline of a line: its origin, its unit direction and its angle.

    Words are placed along this one axis instead of at their own box centres, which is what keeps
    a line straight; the perpendicular median absorbs boxes that sit a little high or low.
    """
    angle = float(np.median([word.angle for word in words]))
    theta = math.radians(angle)
    ux, uy = math.cos(theta), -math.sin(theta)
    x0, y0 = words[0].x, words[0].y
    across = float(np.median([(word.x - x0) * -uy + (word.y - y0) * ux for word in words]))
    return x0 + across * -uy, y0 + across * ux, ux, uy, angle


def _place_words(offsets: list[float], widths: list[float], space: float, budget: float) -> tuple[list[float], float]:
    """Keep every word where it was detected, unless that would run it into its neighbour.

    A word only moves when the word before it needs the room, and if the line then overruns the
    space it was detected in, the whole line is condensed by a single factor - so words never
    overlap, and the ones that had to give way still share the size of the rest of the page.
    """

    def run(squeeze: float) -> list[float]:
        cursor, placed = -math.inf, []
        for offset, width in zip(offsets, widths):
            start = max(offset, cursor)
            placed.append(start)
            cursor = start + squeeze * (width + space)
        return placed

    placed, squeeze = run(1.0), 1.0
    for _ in range(3):  # pushes shrink as the line condenses, so a few passes converge
        end = placed[-1] + squeeze * widths[-1]
        if end <= budget or budget <= 0:
            break
        squeeze = max(squeeze * budget / end, 0.5)
        placed = run(squeeze)
    return placed, squeeze


def _synthesize_line(
    response: Image.Image,
    words: list[_Word],
    font_size: int,
    font_family: str | None,
    min_font_size: int,
    text_color: tuple[int, int, int],
) -> None:
    """Draw the words of one line at one font size, spaced so that none can overlap the next."""
    ox, oy, ux, uy, angle = _line_axis(words)
    along = sorted(((word.x - ox) * ux + (word.y - oy) * uy, word) for word in words)
    offsets = [offset for offset, _ in along]
    words = [word for _, word in along]
    budget = max(offset + word.width for offset, word in along)

    for _ in range(2):
        font = _cached_font(font_family, font_size)
        squeezes = [max(min(1.0, word.width / _text_width(font, word.value)), 0.5) for word in words]
        widths = [squeeze * _text_width(font, word.value) for squeeze, word in zip(squeezes, words)]
        placed, line_squeeze = _place_words(offsets, widths, 1.0, budget)
        # A line that would have to be condensed to less than half is not crowded but mis-sized:
        # a smaller face keeps it readable instead of squashing the glyphs to nothing.
        if line_squeeze > 0.5 or font_size <= min_font_size:
            break
        font_size = max(int(font_size * 2 * line_squeeze), min_font_size)

    d = ImageDraw.Draw(response)
    for word, offset, squeeze in zip(words, placed, squeezes):
        x, y = round(ox + offset * ux), round(oy + offset * uy)
        squeeze *= line_squeeze
        if abs(angle) > 3 or squeeze < 1.0:
            _paste_word(response, word.value, font, (x, y), text_color, angle, squeeze)
        else:
            # "lm" anchor: vertically centered on the baseline, no ascender-offset drift
            _draw_word(d, (x, y), word.value, font, text_color, anchor="lm")


def _synthesize_value(
    response: Image.Image,
    entry: dict[str, Any],
    polygon: list[tuple[float, float]],
    angle: float,
    w: int,
    h: int,
    font_size: int,
    font_family: str | None,
    text_color: tuple[int, int, int],
) -> None:
    """Draw a single value (a word entry, or a KIE prediction) inside its own box."""
    text = str(entry["value"])
    # Measure along the text direction, not on the axis-aligned bounding box
    box_w = max(int(round(math.hypot((polygon[1][0] - polygon[0][0]) * w, (polygon[1][1] - polygon[0][1]) * h))), 1)
    font = _cached_font(font_family, font_size)
    # Anchor on the middle of the leading edge, like the "lm" anchor of flat text
    x = int(round(w * (polygon[0][0] + polygon[3][0]) / 2))
    y = int(round(h * (polygon[0][1] + polygon[3][1]) / 2))
    squeeze = min(1.0, box_w / _text_width(font, text))  # stay inside the box, do not run into the next field
    if abs(angle) > 3 or squeeze < 1.0:
        _paste_word(response, text, font, (x, y), text_color, angle, squeeze)
    else:
        _draw_word(ImageDraw.Draw(response), (x, y), text, font, text_color, anchor="lm")


def _draw_confidence(
    response: Image.Image,
    entry: dict[str, Any],
    box: tuple[int, int, int, int],
    font_family: str | None,
) -> None:
    xmin, ymin, xmax, ymax = box
    box_width, box_height = max(xmax - xmin, 1), max(ymax - ymin, 1)
    confidences = [
        float(word["confidence"])
        for word in entry.get("words") or []
        if isinstance(word, dict)
        and isinstance(word.get("confidence"), int | float)
        and math.isfinite(float(word["confidence"]))
    ]
    confidence = entry.get("confidence")
    if not (isinstance(confidence, int | float) and math.isfinite(float(confidence))):
        confidence = float(np.mean(confidences)) if confidences else 1.0
    confidence = min(max(float(confidence), 0.0), 1.0)
    p = int(255 * confidence)
    color = (255 - p, 0, p)

    d = ImageDraw.Draw(response)
    d.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
    # Scale the confidence label with the box instead of a hardcoded size
    prob_font = _cached_font(font_family, max(min(box_height // 2, 20), 10))
    prob_text = f"{confidence:.2f}"
    prob_text_width, prob_text_height = prob_font.getbbox(prob_text)[2:4]
    prob_x_offset = (box_width - prob_text_width) // 2
    prob_y_offset = max(0, ymin - prob_text_height - 2)
    _draw_word(d, (int(xmin + prob_x_offset), int(prob_y_offset)), prob_text, prob_font, color, anchor="lt")


def _entry_geometry(
    entry: dict[str, Any], w: int, h: int
) -> tuple[list[tuple[float, float]], float, tuple[int, int, int, int]] | None:
    geometry = _points(entry.get("geometry"))
    if geometry is None:
        return None
    if len(geometry) == 2:
        (xmin_r, ymin_r), (xmax_r, ymax_r) = geometry
        polygon = [(xmin_r, ymin_r), (xmax_r, ymin_r), (xmax_r, ymax_r), (xmin_r, ymax_r)]
        angle = 0.0
    else:
        polygon, angle = geometry, _polygon_angle(geometry, w, h)
        _warn_rotation_once()  # pragma: no cover
    x_coords, y_coords = zip(*polygon)
    box = (
        int(round(w * min(x_coords))),
        int(round(h * min(y_coords))),
        int(round(w * max(x_coords))),
        int(round(h * max(y_coords))),
    )
    return polygon, angle, box


def _entry_font_size(
    entry: dict[str, Any],
    words: list[_Word],
    polygon: list[tuple[float, float]],
    w: int,
    h: int,
    font_family: str | None,
    min_font_size: int,
    max_font_size: int,
) -> int:
    if words:
        return _fit_line_font_size(words, font_family, min_font_size, max_font_size)
    box_w = max(int(round(math.hypot((polygon[1][0] - polygon[0][0]) * w, (polygon[1][1] - polygon[0][1]) * h))), 1)
    box_h = max(int(round(math.hypot((polygon[2][0] - polygon[1][0]) * w, (polygon[2][1] - polygon[1][1]) * h))), 1)
    return _fit_font_size(str(entry["value"]), box_w, box_h, font_family, min_font_size, max_font_size)


def _page_size(page: dict[str, Any]) -> tuple[int, int]:
    try:
        h, w = (int(round(float(value))) for value in page["dimensions"])
    except Exception as exc:
        raise ValueError(f"invalid page 'dimensions': {page.get('dimensions')!r}") from exc
    if h <= 0 or w <= 0:
        raise ValueError(f"page dimensions must be positive, got {(h, w)}")
    if Image.MAX_IMAGE_PIXELS is not None and h * w > Image.MAX_IMAGE_PIXELS:
        raise ValueError(f"page dimensions {(h, w)} exceed Pillow's MAX_IMAGE_PIXELS safety limit")
    return h, w


def _render(
    page: dict[str, Any],
    entries: list[dict[str, Any]],
    draw_proba: bool,
    font_family: str | None,
    min_font_size: int,
    max_font_size: int,
    background_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> np.ndarray:
    """Render entries onto a blank page at one harmonized set of font sizes."""
    h, w = _page_size(page)
    response = Image.new("RGB", (w, h), color=background_color)

    prepared = []
    for entry in entries:
        geometry = _entry_geometry(entry, w, h)
        if geometry is None:
            continue
        polygon, angle, box = geometry
        words = _entry_words(entry, w, h)
        if not words and not str(entry.get("value", "")).strip():
            if draw_proba:
                _draw_confidence(response, entry, box, font_family)
            continue
        try:
            size = _entry_font_size(entry, words, polygon, w, h, font_family, min_font_size, max_font_size)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"Could not size entry: {exc}")
            continue
        prepared.append((entry, polygon, angle, box, words, size))

    sizes = _harmonize([size for *_, size in prepared]) if prepared else []
    for (entry, polygon, angle, box, words, _), size in zip(prepared, sizes):
        try:
            if words:
                _synthesize_line(response, words, size, font_family, min_font_size, text_color)
            else:
                _synthesize_value(response, entry, polygon, angle, w, h, size, font_family, text_color)
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"Could not render entry: {exc}")
        if draw_proba:
            _draw_confidence(response, entry, box, font_family)

    return np.array(response, dtype=np.uint8)


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
    lines = [
        line
        for block in page.get("blocks") or []
        if isinstance(block, dict)
        for line in block.get("lines") or []
        if isinstance(line, dict)
    ]
    return _render(page, lines, draw_proba, font_family, min_font_size, max_font_size, background_color, text_color)


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
    predictions = [
        prediction
        for predictions in (page.get("predictions") or {}).values()
        for prediction in predictions or []
        if isinstance(prediction, dict)
    ]
    return _render(
        page, predictions, draw_proba, font_family, min_font_size, max_font_size, background_color, text_color
    )

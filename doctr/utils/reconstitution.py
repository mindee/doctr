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
    """A word to render, with its text, position, size and rotation."""

    value: str
    x: int
    y: int
    width: int
    height: int
    angle: float  # degrees, counter-clockwise


@lru_cache(maxsize=256)
def _cached_font(font_family: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Memoized font loader: avoids re-reading the font file for every word."""
    try:
        return get_font(font_family, max(font_size, 1))
    except Exception:  # pragma: no cover
        logging.warning(f"Could not load font '{font_family}', falling back to the default font")
        return get_font(None, max(font_size, 1))


@lru_cache(maxsize=1)
def _warn_rotation_once() -> None:  # pragma: no cover
    # lru_cache is thread-safe "warn once" semantics without a mutable global
    logging.warning("Polygons with larger rotations may lead to slightly inaccurate rendering")


def _points(geometry: Any) -> list[tuple[float, float]] | None:
    """Validate a geometry and return its points, or None if it is unusable."""
    try:
        points = [(float(x), float(y)) for x, y in geometry]
    except (TypeError, ValueError):
        return None
    if len(points) not in (2, 4) or not all(math.isfinite(v) for point in points for v in point):
        return None
    return points


def _polygon_angle(polygon: list[tuple[float, float]], w: int, h: int) -> float:
    """Estimate the rotation angle (degrees, counter-clockwise) from the top edge of a 4-point polygon."""
    (x0, y0), (x1, y1) = polygon[0], polygon[1]
    return -math.degrees(math.atan2((y1 - y0) * h, (x1 - x0) * w))


def _text_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str, stroke: int = 0) -> int:
    """Width from the drawing origin to the right edge of the ink, so the left bearing counts."""
    bbox = font.getbbox(text)
    return max(math.ceil(bbox[2]) + 2 * stroke, 1)


def _text_height(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    """Height from the top of the ink to the bottom, so the ascender counts."""
    bbox = font.getbbox(text)
    return max(int(bbox[3]) - int(bbox[1]), 1)


def _text_vspan(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str) -> int:
    """Ascender-to-descender span: the vertical extent the "lm" anchor is centered on."""
    try:
        ascent, descent = font.getmetrics()  # type: ignore[union-attr]
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
    stroke: int = 0,
) -> None:
    """Draw a word, falling back to ASCII if the font cannot render it.

    `stroke` outlines the glyphs in their own colour, which is how a heading is set in bold without
    a second font file
    """
    try:
        try:
            d.text(xy, text, font=font, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=fill)
        except UnicodeEncodeError:
            d.text(xy, anyascii(text), font=font, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=fill)
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
    stroke: int = 0,
) -> None:
    """Render a word on a transparent patch, condense and rotate it, then paste it."""
    pad = 2 + stroke
    patch = Image.new(
        "RGBA",
        (_text_width(font, text, stroke) + 2 * pad, _text_vspan(font, text) + 2 * pad + 2 * stroke),
        (0, 0, 0, 0),
    )
    _draw_word(ImageDraw.Draw(patch), (pad, pad + stroke), text, font, fill, anchor="la", stroke=stroke)
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


def _tilt(words: list[_Word]) -> float | None:
    """Direction of a line in degrees, read from where its words sit, or None if they disagree."""
    if len(words) < 4:
        return None
    xs = np.array([word.x for word in words], dtype=float)
    ys = np.array([word.y for word in words], dtype=float)
    height = float(np.median([word.height for word in words]))
    dx, dy = xs[:, None] - xs[None, :], ys[:, None] - ys[None, :]
    spread = np.abs(dx) > height  # pairs too close together only measure box noise
    if not spread.any():
        return None
    slopes = dy[spread] / dx[spread]
    slope = float(np.median(slopes))
    if float(np.median(np.abs(slopes - slope))) > max(0.5 * abs(slope), 0.02):
        return None
    return -math.degrees(math.atan(slope))


def _unrotate_box(word: _Word, angle: float) -> _Word:
    """Recover the text extent and leading-edge anchor of a word from its axis-aligned box."""
    s, c = abs(math.sin(math.radians(angle))), abs(math.cos(math.radians(angle)))
    det = c * c - s * s
    if det < 0.2:  # past ~39 degrees the inversion stops being trustworthy
        return word
    width = (word.width * c - word.height * s) / det
    height = (word.height * c - word.width * s) / det
    if width < 1 or height < 1:
        return word
    # The anchor moves from the middle of the box edge to the middle of the leading edge of the text
    x = word.x + height * s / 2
    y = word.y - word.height / 2 + height * c / 2 if angle < 0 else word.y + word.height / 2 - height * c / 2
    return word._replace(x=round(x), y=round(y), width=round(width), height=round(height), angle=angle)


def _space_width(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> float:
    """Half of the font's own space advance: the least gap that still reads as a word break."""
    try:
        return max(float(font.getlength(" ")) / 2, 1.0)
    except Exception:  # pragma: no cover - a bitmap font may not measure a space
        return max(_text_width(font, "n") / 2, 1.0)


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
    # An upright box carries no angle of its own, so a tilted line has to be recognised from where
    # its words sit before anything can be sized off boxes that the tilt has inflated
    tilt = _tilt(words) if words and not any(word.angle for word in words) else None
    return [_unrotate_box(word, tilt) for word in words] if tilt and abs(tilt) >= 3 else words


def _line_axis(words: list[_Word]) -> tuple[float, float, float, float, float]:
    """The baseline of a line: its origin, its unit direction and its angle."""
    # Where the words sit pins the direction down far more tightly than their own corners do,
    # so the box angles are only a fallback for lines too short to read a trend from
    angle = _tilt(words)
    if angle is None:
        angle = _median_angle(words)
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


def _median_angle(words: list[_Word]) -> float:
    """Median of the word angles, folded so they all lie within a quarter turn of zero."""
    return float(np.median([(word.angle + 90) % 180 - 90 for word in words]))


def _split_rows(words: list[_Word]) -> list[list[_Word]]:
    """Split a group of words into the separate baselines they actually sit on."""
    if len(words) < 2:
        return [words]
    theta = math.radians(_median_angle(words))
    vx, vy = math.sin(theta), math.cos(theta)  # across the text direction
    tolerance = 0.8 * float(np.median([word.height for word in words]))
    ordered = sorted(words, key=lambda word: word.x * vx + word.y * vy)
    rows, current = [], [ordered[0]]
    for previous, word in zip(ordered, ordered[1:]):
        if (word.x - previous.x) * vx + (word.y - previous.y) * vy > tolerance:
            rows.append(current)
            current = []
        current.append(word)
    rows.append(current)
    return rows


def _place_word(
    response: Image.Image,
    d: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    position: tuple[int, int],
    fill: tuple[int, int, int],
    angle: float = 0.0,
    squeeze: float = 1.0,
    stroke: int = 0,
) -> None:
    """Put one word on the page: straight onto it when it can be, on a patch when it cannot."""
    if abs(angle) > 3 or squeeze < 1.0:
        _paste_word(response, text, font, position, fill, angle, squeeze, stroke)
    else:
        # "lm" anchor: vertically centered on the baseline, no ascender-offset drift
        _draw_word(d, position, text, font, fill, anchor="lm", stroke=stroke)


def _synthesize_line(
    response: Image.Image,
    words: list[_Word],
    font_size: int,
    font_family: str | None,
    min_font_size: int,
    text_color: tuple[int, int, int],
    bold: bool = False,
) -> None:
    """Draw a line, as one row per baseline its words turn out to share."""
    for row in _split_rows(words):
        _synthesize_row(response, row, font_size, font_family, min_font_size, text_color, bold)


def _synthesize_row(
    response: Image.Image,
    words: list[_Word],
    font_size: int,
    font_family: str | None,
    min_font_size: int,
    text_color: tuple[int, int, int],
    bold: bool = False,
) -> None:
    """Draw the words of one row at one font size, spaced so that none can overlap the next."""
    ox, oy, ux, uy, angle = _line_axis(words)
    along = sorted(((word.x - ox) * ux + (word.y - oy) * uy, word) for word in words)
    offsets = [offset for offset, _ in along]
    words = [word for _, word in along]
    budget = max(offset + word.width for offset, word in along)

    for _ in range(2):
        font = _cached_font(font_family, font_size)
        # A bold face is set by outlining the glyphs, which widens them: the layout has to allow
        # for that or the extra weight would spill over the following word
        stroke = max(round(font_size / 32), 1) if bold else 0
        squeezes = [max(min(1.0, word.width / _text_width(font, word.value, stroke)), 0.5) for word in words]
        widths = [squeeze * _text_width(font, word.value, stroke) for squeeze, word in zip(squeezes, words)]
        # The boxes cannot supply the spacing (a detector dilates them), so the layout keeps at
        # least a readable gap of its own between one word and the next
        placed, line_squeeze = _place_words(offsets, widths, _space_width(font), budget)
        # A line that would have to be condensed to less than half is not crowded but mis-sized:
        # a smaller face keeps it readable instead of squashing the glyphs to nothing.
        if line_squeeze > 0.5 or font_size <= min_font_size:
            break
        font_size = max(int(font_size * 2 * line_squeeze), min_font_size)

    d = ImageDraw.Draw(response)
    for word, offset, squeeze in zip(words, placed, squeezes):
        x, y = round(ox + offset * ux), round(oy + offset * uy)
        _place_word(response, d, word.value, font, (x, y), text_color, angle, squeeze * line_squeeze, stroke)


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
    bold: bool = False,
) -> None:
    """Draw a single value (a word entry, or a KIE prediction) inside its own box."""
    text = str(entry["value"])
    # Measure along the text direction, not on the axis-aligned bounding box
    box_w = max(int(round(math.hypot((polygon[1][0] - polygon[0][0]) * w, (polygon[1][1] - polygon[0][1]) * h))), 1)
    font = _cached_font(font_family, font_size)
    # Anchor on the middle of the leading edge, like the "lm" anchor of flat text
    x = int(round(w * (polygon[0][0] + polygon[3][0]) / 2))
    y = int(round(h * (polygon[0][1] + polygon[3][1]) / 2))
    stroke = max(round(font_size / 64), 1) if bold else 0
    # stay inside the box, do not run into the next field
    squeeze = min(1.0, box_w / _text_width(font, text, stroke))
    _place_word(response, ImageDraw.Draw(response), text, font, (x, y), text_color, angle, squeeze, stroke)


def _draw_confidence(
    response: Image.Image,
    entry: dict[str, Any],
    box: tuple[int, int, int, int],
    font_family: str | None,
) -> None:
    """Outline the entry and label it with its confidence. Blue: p=1, red: p=0."""
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


def _region_kind(kind: str) -> str:
    """Normalize a layout class name so 'Section-header', 'section header' and 'SECTION_HEADER' match."""
    return "".join(character for character in kind.lower() if character.isalnum())


def _region_type(regions: list[tuple[tuple[int, int, int, int], str]], point: tuple[int, int]) -> str:
    """Type of the smallest layout region holding a point, or "" if it falls outside them all."""
    holding = [(box, kind) for box, kind in regions if box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]]
    if not holding:
        return ""
    box, kind = min(holding, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
    return kind


def _draw_region(
    response: Image.Image,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str = "",
    font_family: str | None = None,
) -> None:
    """Outline a region of the page and, if it is roomy enough, name it."""
    xmin, ymin, xmax, ymax = box
    d = ImageDraw.Draw(response)
    d.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
    if label:
        size = _fit_font_size(label, max(xmax - xmin, 1), max((ymax - ymin) // 4, 1), font_family, 8, 20)
        font = _cached_font(font_family, size)
        _draw_word(d, ((xmin + xmax) // 2, (ymin + ymax) // 2), label, font, color, anchor="mm")


def _entry_geometry(
    entry: dict[str, Any], w: int, h: int
) -> tuple[list[tuple[float, float]], float, tuple[int, int, int, int]] | None:
    """Normalize an entry geometry to a 4-point polygon, its angle and its pixel bounding box."""
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
    """The size this entry would like, before it is harmonized with the rest of the page."""
    if words:
        return _fit_line_font_size(words, font_family, min_font_size, max_font_size)
    box_w = max(int(round(math.hypot((polygon[1][0] - polygon[0][0]) * w, (polygon[1][1] - polygon[0][1]) * h))), 1)
    box_h = max(int(round(math.hypot((polygon[2][0] - polygon[1][0]) * w, (polygon[2][1] - polygon[1][1]) * h))), 1)
    return _fit_font_size(str(entry["value"]), box_w, box_h, font_family, min_font_size, max_font_size)


def _page_size(page: dict[str, Any]) -> tuple[int, int]:
    """Validate the page dimensions before allocating the canvas."""
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
    draw_placeholders: bool,
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
        except Exception as exc:
            logging.warning(f"Could not size entry: {exc}")
            continue
        prepared.append((entry, polygon, angle, box, words, size))

    # The grid of a table, and the outline of a region holding no text of its own - a picture, a
    # formula, a region the predictor was told to ignore
    faint: tuple[int, int, int] = tuple((2 * back + fore) // 3 for back, fore in zip(background_color, text_color))  # type: ignore[assignment]
    filled: list[tuple[int, int]] = []
    if draw_placeholders:
        for _, _, _, box, entry_words, _ in prepared:
            filled.extend(
                [(word.x, word.y) for word in entry_words] or [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)]
            )
        for table in page.get("tables") or []:
            if not isinstance(table, dict):
                continue
            for region in [table, *(table.get("cells") or [])]:
                geometry = _entry_geometry(region, w, h) if isinstance(region, dict) else None
                if geometry is not None:
                    _draw_region(response, geometry[2], faint)
    regions = []
    for region in page.get("layout") or []:
        geometry = _entry_geometry(region, w, h) if isinstance(region, dict) else None
        if geometry is None:
            continue
        regions.append((geometry[2], str(region.get("type", "")).strip()))
        xmin, ymin, xmax, ymax = geometry[2]
        if draw_placeholders and not any(xmin <= x <= xmax and ymin <= y <= ymax for x, y in filled):
            _draw_region(response, geometry[2], faint, str(region.get("type", "")).strip(), font_family)

    sizes = [size for *_, size in prepared]
    # A table cell is a structural box, not an ink box: it is as tall as its row, padding and all,
    # so letting its text fill it renders a table twice the size of the page it sits on. The text
    # a cell holds came off the same page, so the size the rest of the page uses is the better bet.
    body = [size for (entry, *_), size in zip(prepared, sizes) if "row_start" not in entry]
    if body:
        sizes = [
            min(size, int(np.median(body))) if "row_start" in entry else size
            for (entry, *_), size in zip(prepared, sizes)
        ]
    # Harmonizing inside a region rather than across the whole page is what keeps a section header
    # its own size: it is often only a little larger than the body text, and a page-wide median
    # would swallow it. Entries outside every region keep harmonizing with each other as before.
    grouped: dict[str, list[int]] = {}
    for index, (_, _, _, box, _, _) in enumerate(prepared):
        centre = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        grouped.setdefault(_region_type(regions, centre), []).append(index)
    for indices in grouped.values():
        for index, size in zip(indices, _harmonize([sizes[index] for index in indices])):
            sizes[index] = size

    for (entry, polygon, angle, box, words, _), size in zip(prepared, sizes):
        centre = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        kind = _region_kind(_region_type(regions, centre))
        bold = kind in ("title", "sectionheader")
        try:
            if words:
                _synthesize_line(response, words, size, font_family, min_font_size, text_color, bold)
            else:
                _synthesize_value(response, entry, polygon, angle, w, h, size, font_family, text_color, bold)
        except Exception as exc:
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
    draw_placeholders: bool = False,
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
        draw_placeholders: if True, draw the grid of the recognized tables and outline the layout
            regions holding no text of their own

    Returns:
        the synthesized page
    """
    entries = [
        line
        for block in page.get("blocks") or []
        if isinstance(block, dict)
        for line in block.get("lines") or []
        if isinstance(line, dict)
    ]
    # Words falling in a table are regrouped into its cells and removed from the blocks, so the
    # cells are the only place that text still exists
    entries += [
        cell
        for table in page.get("tables") or []
        if isinstance(table, dict)
        for cell in table.get("cells") or []
        if isinstance(cell, dict)
    ]
    return _render(
        page,
        entries,
        draw_proba,
        font_family,
        min_font_size,
        max_font_size,
        background_color,
        text_color,
        draw_placeholders,
    )


def synthesize_kie_page(
    page: dict[str, Any],
    draw_proba: bool = False,
    font_family: str | None = None,
    min_font_size: int = 8,
    max_font_size: int = 50,
    background_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0),
    draw_placeholders: bool = False,
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
        draw_placeholders: if True, draw the grid of the recognized tables and outline the layout
            regions holding no text of their own

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
        page,
        predictions,
        draw_proba,
        font_family,
        min_font_size,
        max_font_size,
        background_color,
        text_color,
        draw_placeholders,
    )

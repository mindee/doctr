# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import platform
from functools import lru_cache

from PIL import ImageFont

__all__ = ["get_font", "get_font_candidates"]

logger = logging.getLogger(__name__)

_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Linux": (
        "DejaVuSans.ttf",
        "NotoSans-Regular.ttf",
        "LiberationSans-Regular.ttf",
        "FreeSans.ttf",
        "FreeMono.ttf",  # legacy default
    ),
    "Darwin": (
        "Arial Unicode.ttf",
        "Helvetica.ttc",
        "Arial.ttf",  # legacy default
    ),
    "Windows": (
        "arial.ttf",  # legacy default
        "segoeui.ttf",
        "tahoma.ttf",
    ),
}


def _is_available(font_family: str) -> bool:
    try:
        _ = ImageFont.truetype(font_family, 10)
    except OSError:
        return False
    return True


def get_font_candidates() -> tuple[str, ...]:
    """Lists the recommended fonts which are installed on this system

    Returns:
        the available font families, by order of preference
    """
    candidates = _FONT_CANDIDATES.get(platform.system(), _FONT_CANDIDATES["Linux"])
    return tuple(family for family in candidates if _is_available(family))


@lru_cache(maxsize=1)
def _resolve_default_font_family() -> str | None:
    """Find the first available candidate font for this platform."""
    candidates = get_font_candidates()
    return candidates[0] if candidates else None


def get_font(font_family: str | None = None, font_size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolves a compatible ImageFont for the system

    Args:
        font_family: the font family (or path to a font file) to use. If None,
            the best available system font is picked automatically.
        font_size: the size of the font upon rendering

    Returns:
        the Pillow font
    """
    if font_family is not None:
        return ImageFont.truetype(font_family, font_size)

    default_family = _resolve_default_font_family()
    if default_family is not None:
        return ImageFont.truetype(default_family, font_size)

    # Last resort: Pillow's built-in font.
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:  # pragma: no cover
        logger.warning(
            "Unable to load any recommended font family. Loading default PIL font, "
            "font size issues may be expected. "
            "To prevent this, it is recommended to specify the value of 'font_family'."
        )
        return ImageFont.load_default()

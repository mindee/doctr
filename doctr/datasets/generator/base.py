# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import random
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from doctr.io.image import tensor_from_pil
from doctr.utils import Sample
from doctr.utils.fonts import get_font, get_font_candidates

from ..datasets import AbstractDataset

# Size the synthetic samples are rendered at, and therefore the size the vocabulary is
# probed at: whether a glyph carries ink is a property of the rasterized outline.
DEFAULT_FONT_SIZE = 32
# Unicode non-characters: permanently unassigned, so a font never maps them and always
# falls back to ".notdef". Rendering them is how we learn what that fallback looks like.
NON_CHARACTERS = ("\ufffe", "\U0010ffff")


@lru_cache(maxsize=32)
def _load_font(font_family: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return get_font(font_family, font_size)


def _glyph_ink(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, char: str) -> tuple[tuple[int, int], bytes] | None:
    """Rasterized glyph of a character, None if it is drawn without ink."""
    mask = font.getmask(char, mode="L")
    return (mask.size, bytes(mask)) if mask.getbbox() is not None else None


@lru_cache(maxsize=32)
def _notdef_ink(font_family: str | None, font_size: int) -> frozenset[tuple[tuple[int, int], bytes]]:
    """Glyph a font falls back to for characters it does not cover (the ".notdef" box)."""
    font = _load_font(font_family, font_size)
    return frozenset(ink for char in NON_CHARACTERS if (ink := _glyph_ink(font, char)) is not None)


@lru_cache(maxsize=8192)
def _renders_char(char: str, font_family: str | None, font_size: int) -> bool:
    """Whether a font draws the actual glyph of a character, rather than nothing or a box."""
    ink = _glyph_ink(_load_font(font_family, font_size), char)
    return ink is not None and ink not in _notdef_ink(font_family, font_size)


def _resolve_fonts(font_family: str | list[str] | None) -> list[str | None]:
    font_families: list[str | None] = [*font_family] if isinstance(font_family, list) else [font_family]
    for font in font_families:
        try:
            _ = _load_font(font, DEFAULT_FONT_SIZE)
        except OSError:
            raise ValueError(f"unable to locate font: {font}")
    return font_families


def _fonts_per_char(
    vocab: str, font_families: Sequence[str | None], font_size: int = DEFAULT_FONT_SIZE
) -> dict[str, tuple[str | None, ...]]:
    """Maps each character of the vocab to the fonts able to render it.

    Characters none of the given fonts can render fall back to the system fonts.

    Args:
        vocab: the characters to render
        font_families: the font families to pick from
        font_size: the size the characters are rendered at

    Returns:
        the fonts able to render each character

    Raises:
        ValueError: if a character is rendered by none of the fonts
    """
    # Whitespace is inkless in every font by design, so no font can be said to "cover" it;
    # judging it by ink would reject any vocabulary containing a space.
    mapping = {
        char: tuple(font for font in font_families if char.isspace() or _renders_char(char, font, font_size))
        for char in dict.fromkeys(vocab)
    }
    missing = [char for char, fonts in mapping.items() if not fonts]
    if missing:
        fallbacks = [font for font in get_font_candidates() if font not in font_families]
        mapping.update({
            char: tuple(font for font in fallbacks if _renders_char(char, font, font_size)) for char in missing
        })
        unrenderable = [char for char in missing if not mapping[char]]
        if unrenderable:
            raise ValueError(
                f"the following characters cannot be rendered, neither by the given fonts {list(font_families)} "
                f"nor by the system fonts {fallbacks}: "
                f"{', '.join(f'{char!r} (U+{ord(char):04X})' for char in unrenderable)}. "
                "They are drawn blank or as a '.notdef' box, please provide font(s) covering the whole vocab."
            )
    return mapping


def synthesize_text_img(
    text: str,
    font_size: int = DEFAULT_FONT_SIZE,
    font_family: str | None = None,
    background_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Generate a synthetic text image

    Args:
        text: the text to render as an image
        font_size: the size of the font
        font_family: the font family (has to be installed on your system)
        background_color: background color of the final image
        text_color: text color on the final image

    Returns:
        PIL image of the text

    Raises:
        ValueError: if the text is empty or rendered without ink
    """
    if not text:
        raise ValueError("unable to synthesize an image from an empty text")

    background_color = (0, 0, 0) if background_color is None else background_color
    text_color = (255, 255, 255) if text_color is None else text_color

    font = _load_font(font_family, font_size)
    left, top, right, bottom = font.getbbox(text)
    text_w, text_h = right - left, bottom - top
    if text_w <= 0 or text_h <= 0:
        raise ValueError(f"font {font_family!r} draws no ink for {text!r}, resulting in a zero-dimension image")

    h, w = int(round(1.3 * text_h)), int(round(1.1 * text_w))
    # If single letter, make the image square, otherwise expand to meet the text size
    img_size = (h, w) if len(text) > 1 else (max(h, w), max(h, w))

    img = Image.new("RGB", img_size[::-1], color=background_color)
    d = ImageDraw.Draw(img)

    # Offset so that the text is centered
    text_pos = (int(round((img_size[1] - text_w) / 2)), int(round((img_size[0] - text_h) / 2)))
    # Draw the text
    d.text(text_pos, text, font=font, fill=text_color)
    return img


class _CharacterGenerator(AbstractDataset):
    def __init__(
        self,
        vocab: str,
        num_samples: int,
        cache_samples: bool = False,
        font_family: str | list[str] | None = None,
        img_transforms: Callable[[Any], Any] | None = None,
        sample_transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        self.vocab = vocab
        self._num_samples = num_samples
        self.font_family = _resolve_fonts(font_family)
        self._fonts_per_char = _fonts_per_char(self.vocab, self.font_family)
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms

        self._data: list[Image.Image] = []
        if cache_samples:
            self._data = [
                (synthesize_text_img(char, font_family=font), idx)  # type: ignore[misc]
                for idx, char in enumerate(self.vocab)
                for font in self._fonts_per_char[char]
            ]

    def __len__(self) -> int:
        return self._num_samples

    def _read_sample(self, index: int) -> tuple[Any, int]:
        # Samples are already cached
        if len(self._data) > 0:
            idx = index % len(self._data)
            pil_img, target = self._data[idx]  # type: ignore[misc]
        else:
            target = index % len(self.vocab)
            char = self.vocab[target]
            pil_img = synthesize_text_img(char, font_family=random.choice(self._fonts_per_char[char]))
        img = tensor_from_pil(pil_img)

        return img, target


class _WordGenerator(AbstractDataset):
    def __init__(
        self,
        vocab: str,
        min_chars: int,
        max_chars: int,
        num_samples: int,
        cache_samples: bool = False,
        font_family: str | list[str] | None = None,
        img_transforms: Callable[[Any], Any] | None = None,
        sample_transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        self.vocab = vocab
        self.wordlen_range = (min_chars, max_chars)
        self._num_samples = num_samples
        self.font_family = _resolve_fonts(font_family)
        fonts_per_char = _fonts_per_char(self.vocab, self.font_family)
        # A word is drawn with a single font, so the font is picked first and the word is
        # built from the characters this font can render, weighted by its coverage
        self._vocab_per_font = {
            font: "".join(char for char in self.vocab if font in fonts_per_char[char])
            for font in dict.fromkeys(font for fonts in fonts_per_char.values() for font in fonts)
        }
        self._fonts = list(self._vocab_per_font)
        self._font_weights = [len(vocab_) for vocab_ in self._vocab_per_font.values()]
        self.img_transforms = img_transforms
        self.sample_transforms = sample_transforms

        self._data: list[Image.Image] = []
        if cache_samples:
            self._data = [self._synthesize_sample() for _ in range(num_samples)]  # type: ignore[misc]

    def _generate_string(self, min_chars: int, max_chars: int, vocab: str | None = None) -> str:
        num_chars = random.randint(min_chars, max_chars)
        return "".join(random.choice(self.vocab if vocab is None else vocab) for _ in range(num_chars))

    def _synthesize_sample(self) -> tuple[Image.Image, str]:
        font = random.choices(self._fonts, weights=self._font_weights)[0]
        text = self._generate_string(*self.wordlen_range, vocab=self._vocab_per_font[font])
        return synthesize_text_img(text, font_family=font), text

    def __len__(self) -> int:
        return self._num_samples

    def _read_sample(self, index: int) -> tuple[Any, str]:
        # Samples are already cached
        if len(self._data) > 0:
            pil_img, target = self._data[index]  # type: ignore[misc]
        else:
            pil_img, target = self._synthesize_sample()
        img = tensor_from_pil(pil_img)

        return img, target

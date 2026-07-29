# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from doctr.utils.geometry import extract_crops, extract_rcrops

if TYPE_CHECKING:  # pragma: no cover
    from doctr.io.elements import LayoutElement, Page

__all__ = [
    "IMAGE_FORMATS",
    "IMAGE_MODES",
    "FigureEncoder",
    "crop_layout_region",
    "encode_crop",
    "is_picture_label",
    "is_picture_region",
    "picture_regions",
]

# How the figures detected by the layout model are materialized in the Markdown / AsciiDoc / HTML exports
IMAGE_MODES = ("none", "placeholder", "embedded", "referenced")
IMAGE_FORMATS = ("png", "jpg", "jpeg", "webp")


def is_picture_label(label: str | None) -> bool:
    """Whether a layout label denotes a figure (as opposed to a table or a text region).

    Args:
        label: the layout label to inspect (e.g. a DocLayNet class such as 'Picture' or 'Table')

    Returns:
        True for the float labels that are not tables ('Picture', 'Figure', 'Chart', ...)
    """
    from doctr.models.reading_order import layout_label_role, normalize_layout_label

    return layout_label_role(label) == "float" and normalize_layout_label(label) != "table"


def is_picture_region(region: "LayoutElement") -> bool:
    """Whether a layout region is a figure (as opposed to a table or a text region).

    Args:
        region: the layout region to inspect

    Returns:
        True for the float regions that are not tables ('Picture', 'Figure', 'Chart', ...)
    """
    return is_picture_label(getattr(region, "type", None))


def picture_regions(page: "Page") -> list["LayoutElement"]:
    """The figure regions detected on a page, in the order the layout model returned them.

    Args:
        page: the page to inspect

    Returns:
        the list of picture regions (empty when the page carries no layout)
    """
    return [region for region in (getattr(page, "layout", None) or []) if is_picture_region(region)]


def _pad_geometry(points: np.ndarray, padding: float) -> np.ndarray:
    """Grow a geometry around its center by a relative margin, and clip it back to the page."""
    if padding == 0:
        return points
    center = points.mean(axis=0, keepdims=True)
    return np.clip(center + (points - center) * (1 + 2 * padding), 0, 1)


def crop_layout_region(
    page_img: np.ndarray | None,
    geometry: Any,
    padding: float = 0.0,
) -> np.ndarray | None:
    """Crop the pixels of a layout region out of its page.

    Straight regions are sliced out of the page, rotated ones are de-rotated with a warp (the layout
    polygons are reading-oriented, exactly like the detection ones).

    Args:
        page_img: the page image, as stored on `Page.page`. An empty array (a page restored from a
            JSON export) or None yields None.
        geometry: the region geometry, either a straight ((xmin, ymin), (xmax, ymax)) box or a (4, 2)
            polygon, with coordinates relative to the page size
        padding: relative margin added around the region on each side (0.05 grows it by 5%)

    Returns:
        the cropped image, or None when the page carries no pixels or the region is degenerate (empty or
        smaller than 2x2 pixels)
    """
    if page_img is None or page_img.size == 0:
        return None
    points = np.asarray(geometry, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] not in (2, 4):
        return None
    points = _pad_geometry(points, padding)
    if points.shape[0] == 2:  # straight box
        box = np.array(
            [[points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]], dtype=np.float32
        )
        crops = extract_crops(page_img, box)
    else:  # rotated polygon
        crops = extract_rcrops(page_img, points[None, ...].astype(np.float32))
    if len(crops) == 0 or crops[0].size == 0 or min(crops[0].shape[:2]) < 2:
        return None  # a collapsed region would otherwise yield a 1-pixel image
    return crops[0]


def encode_crop(crop: np.ndarray, image_format: str = "png", quality: int = 95) -> bytes:
    """Encode a crop into an image file format.

    Args:
        crop: the RGB crop to encode (docTR pages are RGB, OpenCV expects BGR)
        image_format: one of 'png', 'jpg'/'jpeg' or 'webp'
        quality: the encoding quality of the lossy formats ('jpg'/'jpeg' and 'webp')

    Returns:
        the encoded image bytes
    """
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"unsupported image format '{image_format}', should be one of {list(IMAGE_FORMATS)}")
    extension = ".jpg" if image_format in ("jpg", "jpeg") else f".{image_format}"
    params: list[int] = []
    if extension == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif extension == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
    array = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if crop.ndim == 3 and crop.shape[2] == 3 else crop
    success, buffer = cv2.imencode(extension, array, params)
    if not success:  # pragma: no cover
        raise RuntimeError(f"failed to encode a figure crop as '{image_format}'")
    return buffer.tobytes()


class FigureEncoder:
    """Turns the figures detected by the layout model into an image source for the text exporters.

    Four modes are available:

    * ``none``: figures are dropped entirely, as they were before this was implemented
    * ``placeholder`` (default): a format-specific comment marks where a figure was detected, without
      touching the pixels
    * ``embedded``: the crop is inlined as a base64 data URI, so the export stays a single file
    * ``referenced``: the crop is written to ``image_dir`` and referenced by a relative path

    >>> from doctr.io import FigureEncoder
    >>> markdown = page.export_as_markdown(images=FigureEncoder("referenced", image_dir="assets"))

    Args:
        mode: one of 'none', 'placeholder', 'embedded' or 'referenced'
        image_dir: the directory the crops are written to (required in 'referenced' mode)
        path_prefix: prepended to the file names in 'referenced' mode, to match the location the export
            is rendered from (e.g. 'assets/' when the Markdown file sits next to the `assets` directory)
        image_format: one of 'png', 'jpg'/'jpeg' or 'webp'
        quality: the encoding quality of the lossy formats
        padding: relative margin added around each region, useful to catch the axis labels of a plot
    """

    def __init__(
        self,
        mode: str = "placeholder",
        image_dir: str | Path | None = None,
        path_prefix: str = "",
        image_format: str = "png",
        quality: int = 95,
        padding: float = 0.0,
    ) -> None:
        if mode not in IMAGE_MODES:
            raise ValueError(f"unsupported image mode '{mode}', should be one of {list(IMAGE_MODES)}")
        if image_format not in IMAGE_FORMATS:
            raise ValueError(f"unsupported image format '{image_format}', should be one of {list(IMAGE_FORMATS)}")
        if mode == "referenced" and image_dir is None:
            raise ValueError("an 'image_dir' is required to export the figures in 'referenced' mode")
        self.mode = mode
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.path_prefix = path_prefix
        self.image_format = image_format
        self.quality = quality
        self.padding = padding
        # The files written so far, in emission order (empty unless the mode is 'referenced')
        self.written: list[Path] = []

    @classmethod
    def resolve(cls, images: "str | FigureEncoder | None") -> "FigureEncoder":
        """Build an encoder from the `images` argument of an export method.

        Args:
            images: an image mode, an already configured encoder, or None (equivalent to 'none')

        Returns:
            the encoder to use
        """
        if isinstance(images, FigureEncoder):
            return images
        return cls(mode="none" if images is None else images)

    @property
    def enabled(self) -> bool:
        """Whether the figures should appear in the export at all"""
        return self.mode != "none"

    @property
    def materializes(self) -> bool:
        """Whether the encoder carries the pixels of the figures (as opposed to marking their position)"""
        return self.mode in ("embedded", "referenced")

    def materializes_on(self, page: "Page") -> bool:
        """Whether the figures of this page will actually carry their pixels.

        The exporters use this to decide whether the text detected inside a figure is redundant: it is
        already visible in the emitted image, but it would be lost with a mere placeholder. A page
        restored from a JSON export carries no pixels, so its inner text must be kept.

        Args:
            page: the page about to be exported

        Returns:
            True when the mode carries the pixels and the page still has an image
        """
        page_img = getattr(page, "page", None)
        return self.materializes and page_img is not None and page_img.size > 0

    def source(self, page: "Page", region: "LayoutElement", index: int) -> str | None:
        """Resolve the image source of a figure.

        Args:
            page: the page the figure belongs to
            region: the picture region to encode
            index: the 1-based index of the figure on the page, used to name the file

        Returns:
            a data URI, a relative path, or None when the pixels are unavailable (which happens in the
            'none' and 'placeholder' modes, and on pages restored from a JSON export)
        """
        if self.mode in ("none", "placeholder"):
            return None
        crop = crop_layout_region(getattr(page, "page", None), region.geometry, self.padding)
        if crop is None:
            return None
        payload = encode_crop(crop, self.image_format, self.quality)
        mime = "jpeg" if self.image_format in ("jpg", "jpeg") else self.image_format
        if self.mode == "embedded":
            from base64 import b64encode

            return f"data:image/{mime};base64,{b64encode(payload).decode('ascii')}"
        extension = "jpg" if mime == "jpeg" else mime
        name = f"page{getattr(page, 'page_idx', 0) + 1}_figure{index}.{extension}"
        assert self.image_dir is not None  # guaranteed by __init__ in 'referenced' mode
        self.image_dir.mkdir(parents=True, exist_ok=True)
        path = self.image_dir / name
        path.write_bytes(payload)
        self.written.append(path)
        return f"{self.path_prefix}{name}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mode='{self.mode}', image_dir={self.image_dir}, "
            f"image_format='{self.image_format}')"
        )

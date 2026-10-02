# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import hashlib
import os
import weakref
from base64 import b64encode
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

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

IMAGE_MODES = ("none", "placeholder", "embedded", "referenced")
IMAGE_FORMATS = ("png", "jpg", "jpeg", "webp")


def is_picture_label(label: str | None) -> bool:
    """Check whether a layout label denotes a figure, i.e. a float that is not a table (e.g. 'Picture', 'Chart')

    Args:
        label: the layout label

    Returns:
        True for a figure label
    """
    from doctr.models.reading_order import layout_label_role, normalize_layout_label

    return layout_label_role(label) == "float" and normalize_layout_label(label) != "table"


def is_picture_region(region: "LayoutElement") -> bool:
    """Check whether a layout region is a figure

    Args:
        region: the layout region

    Returns:
        True for a figure region
    """
    return is_picture_label(getattr(region, "type", None))


def picture_regions(page: "Page") -> list["LayoutElement"]:
    """Return the figure regions of a page, without those nested in a larger figure

    Args:
        page: the page

    Returns:
        the figure regions, in detection order
    """
    from doctr.models.reading_order.base import _to_boxes

    regions = [region for region in (getattr(page, "layout", None) or []) if is_picture_region(region)]
    if len(regions) < 2:
        return regions
    boxes = _to_boxes([region.geometry for region in regions])
    inter_w = np.minimum(boxes[:, None, 2], boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 3], boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    inside = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None) >= 0.5 * areas[:, None]
    # A region is nested when half of it lies in a larger region (or in an earlier one of the same area)
    rank = np.empty(len(regions), dtype=int)
    rank[np.argsort(-areas, kind="stable")] = np.arange(len(regions))
    nested = (inside & (rank[None, :] < rank[:, None])).any(axis=1)
    return [region for region, skip in zip(regions, nested) if not skip]


def _check_padding(padding: float) -> None:
    if not padding >= 0:
        raise ValueError(f"the padding should be a non-negative relative margin, got {padding}")


def _check_quality(quality: int) -> None:
    if not 0 <= quality <= 100:
        raise ValueError(f"the encoding quality should be between 0 and 100, got {quality}")


def _normalize_path_prefix(path_prefix: "str | os.PathLike[str]") -> str:
    """Convert a path prefix to a directory prefix with forward slashes, valid in any link"""
    prefix = os.fspath(path_prefix).replace("\\", "/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _pad_geometry(points: np.ndarray, padding: float) -> np.ndarray:
    """Grow a geometry around its center by a relative margin, clipped to the page"""
    if padding == 0:
        return points
    center = points.mean(axis=0, keepdims=True)
    return np.clip(center + (points - center) * (1 + 2 * padding), 0, 1)


def crop_layout_region(
    page_img: np.ndarray | None,
    geometry: Any,
    padding: float = 0.0,
) -> np.ndarray | None:
    """Crop a layout region out of its page image, de-rotating rotated regions

    Args:
        page_img: the page image, None or empty when the page has no pixels
        geometry: the relative geometry of the region, a straight box or a (4, 2) polygon
        padding: relative margin added on each side

    Returns:
        the crop, or None without pixels or for a region smaller than 2x2 pixels
    """
    _check_padding(padding)
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
        return None
    return crops[0]


def encode_crop(crop: np.ndarray, image_format: str = "png", quality: int = 95) -> bytes:
    """Encode an RGB(A) crop into an image file format

    Args:
        crop: the crop to encode
        image_format: one of 'png', 'jpg', 'jpeg' or 'webp'
        quality: quality of the lossy formats, between 0 and 100

    Returns:
        the encoded image
    """
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"unsupported image format '{image_format}', should be one of {list(IMAGE_FORMATS)}")
    _check_quality(quality)
    extension = ".jpg" if image_format in ("jpg", "jpeg") else f".{image_format}"
    params: list[int] = []
    if extension == ".jpg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif extension == ".webp":
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)]
    array = crop
    # docTR pages are RGB, OpenCV expects BGR
    if crop.ndim == 3 and crop.shape[2] == 3:
        array = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        array = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA)
    success, buffer = cv2.imencode(extension, array, params)
    if not success:  # pragma: no cover
        raise RuntimeError(f"failed to encode a figure crop as '{image_format}'")
    return buffer.tobytes()


class FigureEncoder:
    """Render the figures detected by the layout model in the Markdown, AsciiDoc and HTML exports

    * ``none``: no figures
    * ``placeholder``: a comment marks each figure
    * ``embedded``: each crop is inlined as a base64 data URI
    * ``referenced``: each crop is written to ``image_dir`` and linked by its relative path

    With their pixels ('embedded' or 'referenced'), the text recognized inside the figures is left out of the export.
    Written files are named after the figure position and content, e.g. ``page1_figure2-3fa2b1c9.png``.

    >>> from doctr.io import FigureEncoder
    >>> markdown = page.export_as_markdown(images=FigureEncoder("referenced", image_dir="assets"))

    Args:
        mode: one of 'none', 'placeholder', 'embedded' or 'referenced'
        image_dir: directory the crops are written to (required in 'referenced' mode)
        path_prefix: location of ``image_dir`` relative to the export (``image_dir`` by default)
        image_format: one of 'png', 'jpg', 'jpeg' or 'webp'
        quality: quality of the lossy formats, between 0 and 100
        padding: relative margin added around each region
    """

    def __init__(
        self,
        mode: str = "placeholder",
        image_dir: str | Path | None = None,
        path_prefix: "str | os.PathLike[str] | None" = None,
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
        _check_quality(quality)
        _check_padding(padding)
        self.mode = mode
        self.image_dir = Path(image_dir) if image_dir is not None else None
        if path_prefix is None:
            path_prefix = image_dir if image_dir is not None else ""
        self.path_prefix = _normalize_path_prefix(path_prefix)
        self.image_format = image_format
        self.quality = quality
        self.padding = padding
        # Files written in 'referenced' mode
        self.written: list[Path] = []
        # Encoded sources per page and region id, with weak keys so the cache does not keep the pages alive
        self._sources: "weakref.WeakKeyDictionary[Page, dict[int, tuple[weakref.ref[LayoutElement], str | None]]]" = (
            weakref.WeakKeyDictionary()
        )

    @classmethod
    def resolve(cls, images: "str | FigureEncoder | None") -> "FigureEncoder":
        """Build an encoder from the `images` argument of an export method

        Args:
            images: an image mode, an encoder, or None (same as 'none')

        Returns:
            the encoder
        """
        if isinstance(images, FigureEncoder):
            return images
        if images == "referenced":
            raise ValueError(
                "the 'referenced' mode writes the figures to a directory, pass a configured encoder instead: "
                "images=FigureEncoder('referenced', image_dir='assets')"
            )
        return cls(mode="none" if images is None else images)

    @property
    def enabled(self) -> bool:
        """Whether the figures appear in the export"""
        return self.mode != "none"

    @property
    def materializes(self) -> bool:
        """Whether the figures are exported with their pixels"""
        return self.mode in ("embedded", "referenced")

    def source(self, page: "Page", region: "LayoutElement", index: int) -> str | None:
        """Return the image source of a figure, encoded (and written) on first use

        Args:
            page: the page of the figure
            region: the figure region
            index: 1-based index of the figure on the page, used in the file name

        Returns:
            a data URI or a relative path, None without pixels
        """
        if not self.materializes:
            return None
        cache = self._sources.setdefault(page, {})
        cached = cache.get(id(region))
        # Ids can be recycled: check that the cached one still denotes this region
        if cached is not None and cached[0]() is region:
            return cached[1]
        source = self._encode(page, region, index)
        cache[id(region)] = (weakref.ref(region), source)
        return source

    def _encode(self, page: "Page", region: "LayoutElement", index: int) -> str | None:
        """Crop, encode and, in 'referenced' mode, write a figure"""
        crop = crop_layout_region(getattr(page, "page", None), region.geometry, self.padding)
        if crop is None:
            return None
        payload = encode_crop(crop, self.image_format, self.quality)
        mime = "jpeg" if self.image_format in ("jpg", "jpeg") else self.image_format
        if self.mode == "embedded":
            return f"data:image/{mime};base64,{b64encode(payload).decode('ascii')}"
        extension = "jpg" if mime == "jpeg" else mime
        digest = hashlib.sha256(payload).hexdigest()[:8]
        name = f"page{getattr(page, 'page_idx', 0) + 1}_figure{index}-{digest}.{extension}"
        assert self.image_dir is not None  # set in 'referenced' mode
        self.image_dir.mkdir(parents=True, exist_ok=True)
        path = self.image_dir / name
        path.write_bytes(payload)
        if path not in self.written:
            self.written.append(path)
        return quote(f"{self.path_prefix}{name}", safe="/:")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(mode='{self.mode}', image_dir={self.image_dir}, "
            f"image_format='{self.image_format}')"
        )

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

    A picture lying inside a larger one (a sub-figure or a duplicate detection) is left out: the outer one shows it.

    Args:
        page: the page to inspect

    Returns:
        the list of picture regions (empty when the page carries no layout)
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
    # Rank by decreasing area (first detected on ties): a region is nested when a higher ranked one covers it
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
    """Turn a path prefix into the portable (forward-slash) directory prefix of the referenced figures.

    The prefix locates the image directory as seen from the export, so it is a directory: a missing trailing
    separator is added. Backslashes (Windows separators, e.g. `str(Path("assets"))` on Windows) are turned into
    forward slashes, the only separator Markdown, AsciiDoc and HTML links understand on every platform.
    """
    prefix = os.fspath(path_prefix).replace("\\", "/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


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
        padding: relative margin (non-negative) added around the region on each side (0.05 grows it by 5%)

    Returns:
        the cropped image, or None when the page carries no pixels or the region is degenerate (empty or
        smaller than 2x2 pixels)
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
        return None  # a collapsed region would otherwise yield a 1-pixel image
    return crops[0]


def encode_crop(crop: np.ndarray, image_format: str = "png", quality: int = 95) -> bytes:
    """Encode a crop into an image file format.

    Args:
        crop: the RGB crop to encode (docTR pages are RGB, OpenCV expects BGR)
        image_format: one of 'png', 'jpg'/'jpeg' or 'webp'
        quality: the encoding quality of the lossy formats ('jpg'/'jpeg' and 'webp'), between 0 and 100

    Returns:
        the encoded image bytes
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
    if crop.ndim == 3 and crop.shape[2] == 3:
        array = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        array = cv2.cvtColor(crop, cv2.COLOR_RGBA2BGRA)
    success, buffer = cv2.imencode(extension, array, params)
    if not success:  # pragma: no cover
        raise RuntimeError(f"failed to encode a figure crop as '{image_format}'")
    return buffer.tobytes()


class FigureEncoder:
    """Turns the figures detected by the layout model into an image source for the text exporters.

    Four modes are available:

    * ``none``: figures are left out of the export (they still take part in the reading order, which keeps
      their captions and the text detected inside them in place)
    * ``placeholder`` (default): a format-specific comment marks where a figure was detected, without
      touching the pixels
    * ``embedded``: the crop is inlined as a base64 data URI, so the export stays a single file
    * ``referenced``: the crop is written to ``image_dir`` and referenced by a relative path

    In the 'embedded' and 'referenced' modes, the text recognized inside a figure is dropped from the export (the
    image already shows it), unless that figure could not be cropped. The 'none' and 'placeholder' modes keep it.

    Each figure is encoded (and written) once per encoder. In 'referenced' mode, the file name carries the
    position of the figure and a hash of its content (e.g. ``page1_figure2-3fa2b1c9.png``): several documents can
    share an ``image_dir`` without overwriting each other's figures, and re-exporting a document rewrites the very
    same files.

    >>> from doctr.io import FigureEncoder
    >>> markdown = page.export_as_markdown(images=FigureEncoder("referenced", image_dir="assets"))

    Args:
        mode: one of 'none', 'placeholder', 'embedded' or 'referenced'
        image_dir: the directory the crops are written to (required in 'referenced' mode)
        path_prefix: the location of ``image_dir`` as seen from the export, prepended to the file names in
            'referenced' mode (e.g. 'assets' when the Markdown file sits next to the `assets` directory). A string or
            a path, with or without a trailing separator: Windows backslashes are turned into forward slashes, so
            the links work on every platform. The resulting path is percent-encoded.
        image_format: one of 'png', 'jpg'/'jpeg' or 'webp'
        quality: the encoding quality of the lossy formats, between 0 and 100
        padding: relative margin (non-negative) added around each region, useful to catch the axis labels of a plot
    """

    def __init__(
        self,
        mode: str = "placeholder",
        image_dir: str | Path | None = None,
        path_prefix: "str | os.PathLike[str]" = "",
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
        self.path_prefix = _normalize_path_prefix(path_prefix)
        self.image_format = image_format
        self.quality = quality
        self.padding = padding
        # The files written so far, in emission order (empty unless the mode is 'referenced')
        self.written: list[Path] = []
        # Sources keyed by the ids of the page and region. Weak references tell whether an id still denotes the
        # object it was cached for (ids are recycled once an object dies), without keeping the pages (and their
        # pixels) alive: an encoder reused across a whole corpus only retains the sources themselves.
        self._sources: dict[tuple[int, int], tuple[weakref.ref, weakref.ref, str | None]] = {}

    @classmethod
    def resolve(cls, images: "str | FigureEncoder | None") -> "FigureEncoder":
        """Build an encoder from the `images` argument of an export method.

        Args:
            images: an image mode ('none', 'placeholder' or 'embedded'), an already configured encoder, or None
                (equivalent to 'none'). The 'referenced' mode needs an `image_dir`, so it takes a configured encoder.

        Returns:
            the encoder to use
        """
        if isinstance(images, FigureEncoder):
            return images
        if images == "referenced":
            raise ValueError(
                "the 'referenced' mode writes the figures to a directory, pass a configured encoder instead: "
                "images=FigureEncoder('referenced', image_dir='assets', path_prefix='assets')"
            )
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
        """Whether the figures of this page can carry their pixels.

        A page restored from a JSON export carries no pixels, so its figures fall back to a placeholder. The
        exporters only drop the text detected inside a figure once that figure's source was actually resolved
        (cf. :meth:`source`), since a single region can still fail to be cropped.

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
            'none' and 'placeholder' modes, on pages restored from a JSON export, and on degenerate regions)
        """
        if self.mode in ("none", "placeholder"):
            return None
        key = (id(page), id(region))
        cached = self._sources.get(key)
        if cached is not None and cached[0]() is page and cached[1]() is region:
            return cached[2]
        source = self._encode(page, region, index)
        self._sources[key] = (weakref.ref(page), weakref.ref(region), source)
        # Forget the source along with its page, so a long-lived encoder does not accumulate the data URIs
        weakref.finalize(page, self._sources.pop, key, None)
        return source

    def _encode(self, page: "Page", region: "LayoutElement", index: int) -> str | None:
        """Crop, encode and (in 'referenced' mode) write a figure"""
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
        assert self.image_dir is not None  # guaranteed by __init__ in 'referenced' mode
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

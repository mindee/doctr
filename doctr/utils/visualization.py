# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import mplcursors
from PIL import ImageFont, ImageDraw, Image
from copy import deepcopy
import numpy as np
import cv2
from unidecode import unidecode
from typing import Tuple, List, Dict, Any, Union, Optional

from .common_types import BoundingBox, RotatedBbox
from .fonts import get_font

__all__ = ['visualize_page', 'synthesize_page', 'draw_boxes']


def rect_patch(
    geometry: BoundingBox,
    page_dimensions: Tuple[int, int],
    label: Optional[str] = None,
    color: Tuple[float, float, float] = (0, 0, 0),
    alpha: float = 0.3,
    linewidth: int = 2,
    fill: bool = True,
) -> patches.Rectangle:
    """Create a matplotlib rectangular patch for the element

    Args:
        geometry: bounding box of the element
        page_dimensions: dimensions of the Page
        label: label to display when hovered
        color: color to draw box
        alpha: opacity parameter to fill the boxes, 0 = transparent
        linewidth: line width
        fill: whether the patch should be filled

    Returns:
        a rectangular Patch
    """

    if len(geometry) != 2 or any(not isinstance(elt, tuple) or len(elt) != 2 for elt in geometry):
        raise ValueError("invalid geometry format")

    # Unpack
    height, width = page_dimensions
    (xmin, ymin), (xmax, ymax) = geometry
    # Switch to absolute coords
    xmin, w = xmin * width, (xmax - xmin) * width
    ymin, h = ymin * height, (ymax - ymin) * height

    return patches.Rectangle(
        (xmin, ymin),
        w,
        h,
        fill=fill,
        linewidth=linewidth,
        edgecolor=(*color, alpha),
        facecolor=(*color, alpha),
        label=label,
    )


def polygon_patch(
    geometry: RotatedBbox,
    page_dimensions: Tuple[int, int],
    label: Optional[str] = None,
    color: Tuple[float, float, float] = (0, 0, 0),
    alpha: float = 0.3,
    linewidth: int = 2,
    fill: bool = True,
) -> patches.Polygon:
    """Create a matplotlib polygon patch for the element

    Args:
        geometry: bounding box of the element
        page_dimensions: dimensions of the Page
        label: label to display when hovered
        color: color to draw box
        alpha: opacity parameter to fill the boxes, 0 = transparent
        linewidth: line width
        fill: whether the patch should be filled

    Returns:
        a polygon Patch
    """

    if len(geometry) != 5 or any(not isinstance(elt, float) for elt in geometry):
        raise ValueError("invalid geometry format")

    # Unpack
    height, width = page_dimensions
    x, y, w, h, a = geometry
    # Switch to absolute coords
    x, w = x * width, w * width
    y, h = y * height, h * height
    points = cv2.boxPoints(((x, y), (w, h), a))

    return patches.Polygon(
        points,
        fill=fill,
        linewidth=linewidth,
        edgecolor=(*color, alpha),
        facecolor=(*color, alpha),
        label=label,
    )


def create_obj_patch(
    geometry: Union[BoundingBox, RotatedBbox],
    page_dimensions: Tuple[int, int],
    **kwargs: Any,
) -> patches.Patch:
    """Create a matplotlib patch for the element

    Args:
        geometry: bounding box (straight or rotated) of the element
        page_dimensions: dimensions of the page

    Returns:
        a matplotlib Patch
    """
    if isinstance(geometry, tuple):
        if len(geometry) == 2:
            return rect_patch(geometry, page_dimensions, **kwargs)  # type: ignore[arg-type]
        elif len(geometry) == 5:
            return polygon_patch(geometry, page_dimensions, **kwargs)  # type: ignore[arg-type]

    raise ValueError("invalid geometry format")


def visualize_page(
    page: Dict[str, Any],
    image: np.ndarray,
    words_only: bool = True,
    display_artefacts: bool = True,
    scale: float = 10,
    interactive: bool = True,
    add_labels: bool = True,
    **kwargs: Any,
) -> Figure:
    """Visualize a full page with predicted blocks, lines and words

    Example::
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from doctr.utils.visualization import visualize_page
        >>> from doctr.models import ocr_db_crnn
        >>> model = ocr_db_crnn(pretrained=True)
        >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([[input_page]])
        >>> visualize_page(out[0].pages[0].export(), input_page)
        >>> plt.show()

    Args:
        page: the exported Page of a Document
        image: np array of the page, needs to have the same shape than page['dimensions']
        words_only: whether only words should be displayed
        display_artefacts: whether artefacts should be displayed
        scale: figsize of the largest windows side
        interactive: whether the plot should be interactive
        add_labels: for static plot, adds text labels on top of bounding box
    """
    # Get proper scale and aspect ratio
    h, w = image.shape[:2]
    size = (scale * w / h, scale) if h > w else (scale, h / w * scale)
    fig, ax = plt.subplots(figsize=size)
    # Display the image
    ax.imshow(image)
    # hide both axis
    ax.axis('off')

    if interactive:
        artists: List[patches.Patch] = []  # instantiate an empty list of patches (to be drawn on the page)

    for block in page['blocks']:
        if not words_only:
            rect = create_obj_patch(block['geometry'], page['dimensions'],
                                    label='block', color=(0, 1, 0), linewidth=1, **kwargs)
            # add patch on figure
            ax.add_patch(rect)
            if interactive:
                # add patch to cursor's artists
                artists.append(rect)

        for line in block['lines']:
            if not words_only:
                rect = create_obj_patch(line['geometry'], page['dimensions'],
                                        label='line', color=(1, 0, 0), linewidth=1, **kwargs)
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)

            for word in line['words']:
                rect = create_obj_patch(word['geometry'], page['dimensions'],
                                        label=f"{word['value']} (confidence: {word['confidence']:.2%})",
                                        color=(0, 0, 1), **kwargs)
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)
                elif add_labels:
                    if len(word['geometry']) == 5:
                        text_loc = (
                            int(page['dimensions'][1] * (word['geometry'][0] - word['geometry'][2] / 2)),
                            int(page['dimensions'][0] * (word['geometry'][1] - word['geometry'][3] / 2))
                        )
                    else:
                        text_loc = (
                            int(page['dimensions'][1] * word['geometry'][0][0]),
                            int(page['dimensions'][0] * word['geometry'][0][1])
                        )
                    ax.text(
                        *text_loc,
                        word['value'],
                        size=10,
                        alpha=0.5,
                        color=(0, 0, 1),
                    )

        if display_artefacts:
            for artefact in block['artefacts']:
                rect = create_obj_patch(
                    artefact['geometry'],
                    page['dimensions'],
                    label='artefact',
                    color=(0.5, 0.5, 0.5),
                    linewidth=1,
                    **kwargs
                )
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)

    if interactive:
        # Create mlp Cursor to hover patches in artists
        mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    fig.tight_layout(pad=0.)

    return fig


def synthesize_page(
    page: Dict[str, Any],
    draw_proba: bool = False,
    font_size: int = 13,
    font_family: Optional[str] = None,
) -> np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_size: size of the font, default font = 13
        font_family: family of the font

    Return:
        the synthesized page
    """

    # Draw template
    h, w = page["dimensions"]
    response = 255 * np.ones((h, w, 3), dtype=np.int32)

    # Draw each word
    for block in page["blocks"]:
        for line in block["lines"]:
            for word in line["words"]:
                # Get aboslute word geometry
                (xmin, ymin), (xmax, ymax) = word["geometry"]
                xmin, xmax = int(w * xmin), int(w * xmax)
                ymin, ymax = int(h * ymin), int(h * ymax)

                # White drawing context adapted to font size, 0.75 factor to convert pts --> pix
                font = get_font(font_family, int(0.75 * (ymax - ymin)))
                img = Image.new('RGB', (xmax - xmin, ymax - ymin), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                # Draw in black the value of the word
                try:
                    d.text((0, 0), word["value"], font=font, fill=(0, 0, 0))
                except UnicodeEncodeError:
                    # When character cannot be encoded, use its unidecode version
                    d.text((0, 0), unidecode(word["value"]), font=font, fill=(0, 0, 0))

                # Colorize if draw_proba
                if draw_proba:
                    p = int(255 * word["confidence"])
                    mask = np.where(np.array(img) == 0, 1, 0)
                    proba = np.array([255 - p, 0, p])
                    color = mask * proba[np.newaxis, np.newaxis, :]
                    white_mask = 255 * (1 - mask)
                    img = color + white_mask

                # Write to response page
                response[ymin:ymax, xmin:xmax, :] = np.array(img)

    return response


def draw_boxes(
    boxes: np.ndarray,
    image: np.ndarray,
    color: Optional[Tuple] = None,
    **kwargs
) -> None:
    """Draw an array of relative straight boxes on an image

    Args:
        boxes: array of relative boxes, of shape (*, 4)
        image: np array, float32 or uint8
    """
    h, w = image.shape[:2]
    # Convert boxes to absolute coords
    _boxes = deepcopy(boxes)
    _boxes[:, [0, 2]] *= w
    _boxes[:, [1, 3]] *= h
    _boxes = _boxes.astype(np.int32)
    for box in _boxes.tolist():
        xmin, ymin, xmax, ymax = box
        image = cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color=color if isinstance(color, tuple) else (0, 0, 255),
            thickness=2
        )
    plt.imshow(image)
    plt.plot(**kwargs)

# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import mplcursors
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Union

from .common_types import BoundingBox, RotatedBbox

__all__ = ['visualize_page', 'draw_page']


def create_rect_patch(
    geometry: Union[BoundingBox, RotatedBbox],
    label: str,
    page_dimensions: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.3,
    linewidth: int = 2,
    fill: bool = True,
) -> patches.Patch:
    """Create a matplotlib patch (rectangle) bounding the element

    Args:
        geometry: bounding box of the element
        label: label to display when hovered
        page_dimensions: dimensions of the Page
        color: color to draw box
        alpha: opacity parameter to fill the boxes, 0 = transparent
        linewidth: line width

    Returns:
        a rectangular Patch
    """
    height, width = page_dimensions
    if len(geometry) == 5:
        x, y, w, h, a = geometry  # type: ignore[misc]
        x, w = x * width, w * width
        y, h = y * height, h * height
        points = cv2.boxPoints(((x, y), (w, h), a))
        return patches.Polygon(
            points,
            fill=fill,
            linewidth=linewidth,
            edgecolor=(*color, alpha),
            facecolor=(*color, alpha),
            label=label
        )
    else:
        (xmin, ymin), (xmax, ymax) = geometry  # type: ignore[misc]
        xmin, xmax = xmin * width, xmax * width
        ymin, ymax = ymin * height, ymax * height
        return patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=fill,
            linewidth=linewidth,
            edgecolor=(*color, alpha),
            facecolor=(*color, alpha),
            label=label
        )


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
            rect = create_rect_patch(block['geometry'], 'block', page['dimensions'], (0, 1, 0), linewidth=1, **kwargs)
            # add patch on figure
            ax.add_patch(rect)
            if interactive:
                # add patch to cursor's artists
                artists.append(rect)

        for line in block['lines']:
            if not words_only:
                rect = create_rect_patch(line['geometry'], 'line', page['dimensions'], (1, 0, 0), linewidth=1, **kwargs)
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)

            for word in line['words']:
                rect = create_rect_patch(word['geometry'], f"{word['value']} (confidence: {word['confidence']:.2%})",
                                         page['dimensions'], (0, 0, 1), **kwargs)
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
                rect = create_rect_patch(
                    artefact['geometry'],
                    'artefact',
                    page['dimensions'],
                    (0.5, 0.5, 0.5),  # type: ignore[arg-type]
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


def draw_page(
    page: Dict[str, Any],
    draw_proba: bool = True,
    font_path: str = "fonts/FreeMonoBold.ttf",
) -> np.ndarray:
    """Draw a the content of the element page (OCR response) on a blank page.

    Args:
        page: exported Page object to represent
        draw_proba: if True, draw words in colors to represent confidence. Blue: p=1, red: p=0
        font_path: path to font

    Return:
        A np array (drawn page)
    """
    # Get a drawing context
    h, w = page["dimensions"]
    img = Image.new('RGB', (w, h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Draw each word
    for block in page["blocks"]:
        for line in block["lines"]:
            smoothing = []  # Smooth the size of words on the whole line for a better UX (more esthetic)
            for word in line["words"]:
                # Resize word geometry
                (xmin, ymin), (xmax, ymax) = word["geometry"]
                xmin, xmax = w * xmin, w * xmax
                ymin, ymax = h * ymin, h * ymax

                # Font computation
                smoothing.append(ymax - ymin)  # Update line smoother
                # Convert Pix -> Pts, add line smoothing and offset to reduce boxes to the size of characters.
                font_size = int(.75 * ((ymax - ymin + np.mean(smoothing)) / 2 - 4))
                # Load font
                #fnt = ImageFont.truetype(font_path, font_size)
                fnt = ImageFont.load_default()
                # Draw
                if draw_proba:
                    p = int(255 * word["confidence"])
                    d.text((int(xmin), int(ymin)), word["value"], font=fnt, fill=(255 - p, 0, p))
                else:
                    d.text((int(xmin), int(ymin)), word["value"], font=fnt, fill=(0, 0, 0))

    return np.array(img)

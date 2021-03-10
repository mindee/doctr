# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplcursors
import numpy as np
from typing import Tuple, List, Dict, Any

from .common_types import BoundingBox

__all__ = ['visualize_page']


def create_patch(
    geometry: BoundingBox,
    label: str,
    page_dimensions: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.3,
    linewidth: int = 2,
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
    h, w = page_dimensions
    (xmin, ymin), (xmax, ymax) = geometry
    xmin, xmax = xmin * w, xmax * w
    ymin, ymax = ymin * h, ymax * h
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=True,
        linewidth=linewidth,
        edgecolor=(*color, alpha),
        facecolor=(*color, alpha),
        label=label
    )
    return rect


def visualize_page(
    page: Dict[str, Any],
    image: np.ndarray,
    words_only: bool = True,
) -> None:
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
    """
    # Display the image
    _, ax = plt.subplots()
    ax.imshow(image)
    # hide both axis
    ax.axis('off')

    artists: List[patches.Patch] = []  # instantiate an empty list of patches (to be drawn on the page)

    for block in page['blocks']:
        if not words_only:
            rect = create_patch(block['geometry'], 'block', page['dimensions'], (0, 1, 0), linewidth=1)
            # add patch on figure
            ax.add_patch(rect)
            # add patch to cursor's artists
            artists.append(rect)

        for line in block['lines']:
            if not words_only:
                rect = create_patch(line['geometry'], 'line', page['dimensions'], (1, 0, 0), linewidth=1)
                ax.add_patch(rect)
                artists.append(rect)

            for word in line['words']:
                rect = create_patch(word['geometry'], f"{word['value']} (confidence: {word['confidence']:.2%})",
                                    page['dimensions'], (0, 0, 1))
                ax.add_patch(rect)
                artists.append(rect)

        if not words_only:
            for artefact in block['artefacts']:
                rect = create_patch(artefact['geometry'], 'artefact', page['dimensions'], (0.5, 0.5, 0.5), linewidth=1)
                ax.add_patch(rect)
                artists.append(rect)

    # Create mlp Cursor to hover patches in artists
    mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

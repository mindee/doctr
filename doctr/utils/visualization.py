# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplcursors
import numpy as np
from typing import Union, Tuple, List

from ..documents import Page, Block, Line, Word, Artefact

__all__ = ['visualize_page']


def draw_word(
    word: Word,
    page_dimensions: Tuple[int, int],
    word_color: Tuple[int, int, int],
    alpha: float = 0.3
) -> patches.Patch:
    """Create a matplotlib patch (rectangle) bounding the word

    Args:
        word: Word object to bound
        page_dimensions: dimensions of the Page
        word_color: color to draw box
        alpha: opacity parameter to fill the boxes, 0 = transparent

    Returns:
        a rectangular Patch
    """
    h, w = page_dimensions
    (xmin, ymin), (xmax, ymax) = word.geometry
    xmin, xmax = xmin * w, xmax * w
    ymin, ymax = ymin * h, ymax * h
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=True,
        linewidth=2,
        edgecolor=word_color,
        facecolor=(*word_color, alpha),
        label=f"{word.value} (confidence: {word.confidence:.2%})"
    )
    return rect


def draw_element(
    element: Union[Artefact, Line, Block],
    page_dimensions: Tuple[int, int],
    element_color: Tuple[int, int, int],
    alpha: float = 0.3,
    force_label: bool = True,
) -> patches.Patch:
    """Create a matplotlib patch (rectangle) bounding the element

    Args:
        element: Element (Artefact, Line, Block) to bound
        page_dimensions: dimensions of the Page
        element_color: color to draw box
        alpha: opacity parameter to fill the boxes, 0 = transparent
        force_label: wether to give a label to the patch or not if element = block or line

    Returns:
        a rectangular Patch
    """
    if isinstance(element, Artefact):
        value = "type: {type}, confidence {score}".format(type=element.type, score=element.confidence)
    elif force_label:
        if isinstance(element, Line):
            value = "line"
        elif isinstance(element, Block):
            value = "block"
    else:
        value = None

    h, w = page_dimensions
    (xmin, ymin), (xmax, ymax) = element.geometry
    xmin, xmax = xmin * w, xmax * w
    ymin, ymax = ymin * h, ymax * h
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=True,
        linewidth=1,
        edgecolor=(*element_color, alpha),
        facecolor=(*element_color, alpha),
        label=value
    )
    return rect


def visualize_page(
    page: Page,
    image: np.ndarray,
    words_only: bool = True,
) -> None:
    """Visualize a full page with predicted blocks, lines and words

    Args:
        page: a Page of a Document
        image: np array of the page, needs to have the same shape than page.dimensions
    """
    # Display the image
    _, ax = plt.subplots()
    ax.imshow(image)
    # hide both axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    artists: List[patches.Patch] = []  # instantiate an empty list of patches (to be drawn on the page)

    for block in page.blocks:
        if not words_only:
            rect = draw_element(block, page_dimensions=page.dimensions, element_color=(0, 1, 0))
            # add patch on figure
            ax.add_patch(rect)
            # add patch to cursor's artists
            artists.append(rect)

        for line in block.lines:
            if not words_only:
                rect = draw_element(line, page_dimensions=page.dimensions, element_color=(1, 0, 0))
                ax.add_patch(rect)
                artists.append(rect)

            for word in line.words:
                rect = draw_word(word, page_dimensions=page.dimensions, word_color=(0, 0, 1))
                ax.add_patch(rect)
                artists.append(rect)

    # Create mlp Cursor to hover patches in artists
    mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

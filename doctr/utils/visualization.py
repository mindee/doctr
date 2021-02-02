# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplcursors
from typing import Union, Tuple, List

from doctr.documents import Document, Page, Block, Line, Word, Artefact


def draw_word(
    word: Word,
    page_dimensions: Tuple[int, int],
    color: Tuple[int, int, int],
) -> patches.Patch:
    """Create a matplotlib patch (rectangle) bounding the word

    Args:
        word: Word object to bound
        page_dimensions: dimensions of the Page
        color: color to draw box

    Returns:
        a rectangular Patch
    """
    h, w = page_dimensions
    (xmin, ymin), (xmax, ymax) = word.geometry
    xmin, xmax = xmin * w, xmax * w
    ymin, ymax = ymin * h, ymax * h
    r, g, b = color
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=True,
        linewidth=2,
        edgecolor=(r, g, b),
        facecolor=(r, g, b, 0.3),
        label="value: {value}, confidence {score}".format(value=word.value, score=word.confidence)
    )
    return rect


def draw_element(
    element: Union[Artefact, Line, Block],
    page_dimensions: Tuple[int, int],
    color: Tuple[int, int, int, int],
    label: bool,
) -> patches.Patch:
    """Create a matplotlib patch (rectangle) bounding the element

    Args:
        element: Element (Artefact, Line, Block) to bound
        page_dimensions: dimensions of the Page
        color: color to draw box
        label: wether to give a label to the patch or not if element = block or line

    Returns:
        a rectangular Patch
    """
    if isinstance(element, Artefact):
        value = "type: {type}, confidence {score}".format(type=element.type, score=element.confidence)
    elif label:
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
        edgecolor=color,
        facecolor=color,
        label=value
    )
    return rect


def visualize_page(
    page: Page,
    image: np.ndarray,
) -> None:
    """Visualize a full page with predicted blocks, lines and words

    Args:
        page: a Page of a Document
        image: np array of the page, needs to have the same shape than page.dimensions
    """
    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(image)
    # hide both axis
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    artists: List[patches.Patch] = []  # instantiate an empty list of patches (to be drawn on the page)

    for block in page.blocks:
        rect = draw_element(block, page_dimensions=page.dimensions, color=(0, 1, 0, 0.3), label=True)
        # add patch on figure
        ax.add_patch(rect)
        # add patch to cursor's artists
        artists.append(rect)

        for line in block.lines:
            rect = draw_element(line, page_dimensions=page.dimensions, color=(1, 0, 0, 0.3), label=True)
            ax.add_patch(rect)
            artists.append(rect)

            for word in line.words:
                rect = draw_word(word, page_dimensions=page.dimensions, color=(0, 0, 1))
                ax.add_patch(rect)
                artists.append(rect)

    # Create mlp Cursor to hover patches in artists
    mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    plt.show()

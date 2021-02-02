# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.patches as patches
import mplcursors
from typing import Union, Tuple, List

from doctr.documents.elements import Document, Page, Block, Line, Word


def _mock_words(size=(1., 1.), offset=(0, 0), confidence=0.9):
    return [
        documents.Word("hello", confidence, [
            (offset[0], offset[1]),
            (size[0] / 2 + offset[0], size[1] / 2 + offset[1])
        ]),
        documents.Word("world", confidence, [
            (size[0] / 2 + offset[0], size[1] / 2 + offset[1]),
            (size[0] + offset[0], size[1] + offset[1])
        ])
    ]


def _mock_artefacts(size=(1, 1), offset=(0, 0), confidence=0.8):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        documents.Artefact("qr_code", confidence, [
            (offset[0], offset[1]),
            (sub_size[0] + offset[0], sub_size[1] + offset[1])
        ]),
        documents.Artefact("qr_code", confidence, [
            (sub_size[0] + offset[0], sub_size[1] + offset[1]),
            (size[0] + offset[0], size[1] + offset[1])
        ]),
    ]


def _mock_lines(size=(1, 1), offset=(0, 0)):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        documents.Line(_mock_words(size=sub_size, offset=offset)),
        documents.Line(_mock_words(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1]))),
    ]


def _mock_blocks(size=(1, 1), offset=(0, 0)):
    sub_size = (size[0] / 4, size[1] / 4)
    return [
        documents.Block(
            _mock_lines(size=sub_size, offset=offset),
            _mock_artefacts(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1]))
        ),
        documents.Block(
            _mock_lines(size=sub_size, offset=(offset[0] + 2 * sub_size[0], offset[1] + 2 * sub_size[1])),
            _mock_artefacts(size=sub_size, offset=(offset[0] + 3 * sub_size[0], offset[1] + 3 * sub_size[1])),
        ),
    ]


def _mock_pages(block_size=(1, 1), block_offset=(0, 0)):
    return [
        documents.Page(_mock_blocks(block_size, block_offset), 0, (300, 200),
                       {"value": 0., "confidence": 1.}, {"value": "EN", "confidence": 0.8}),
        documents.Page(_mock_blocks(block_size, block_offset), 1, (500, 1000),
                       {"value": 0.15, "confidence": 0.8}, {"value": "FR", "confidence": 0.7}),
    ]

page = _mock_pages[0]

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
        xmax-xmin,
        ymax-ymin,
        fill=True,
        linewidth=2,
        edgecolor=(r,g,b),
        facecolor=(r,g,b,0.3),
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
    elif label==True:
        if isinstance(element, Line):
            value = "line"
        elif isinstance(element, Block):
            value = "block"
    else:
        value = None

    h, w = page_dimensions
    (xmin, ymin), (xmax, ymax) = word.geometry
    xmin, xmax = xmin * w, xmax * w
    ymin, ymax = ymin * h, ymax * h
    rect = patches.Rectangle(
        (xmin, ymin),
        xmax-xmin,
        ymax-ymin,
        fill=True,
        linewidth=1,
        edgecolor=color,
        facecolor=color,
        label=value
    )
    return rect


def visualize_page(
    page: Page,
) -> None:
    """Visualize a full page with predicted blocks, lines and words

    Args:
        page: a Page of a Document
    """
    # Create figure and axes
    fig, ax = plt.subplots()
    # Read and display the image
    img = image.imread('/home/laptopmindee/Téléchargements/mock_image.jpg')
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    artists: List[patches.Patch] = []  # instantiate an empty list of patches (matplotlib artists to be drawn on the page)

    for block in page.blocks:
        rect = draw_element(block, page_dimensions=page.dimensions, color=(0,1,0,0.3))
        # add patch on figure
        ax.add_patch(rect)
        # add patch to cursor's artists
        artists.append(rect)

        for line in block.lines:
            rect = draw_element(line, page_dimensions=page.dimensions, color=(1,0,0,0.3))
            ax.add_patch(rect)
            artists.append(rect)

            for word in line.words:
                rect = draw_word(word, page_dimensions=page.dimensions, color=(0,0,1))
                ax.add_patch(rect)
                artists.append(rect)
    
    # Create mlp Cursor to hover patches in artists
    mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    plt.show()

visualize_page(page)
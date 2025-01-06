# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
import colorsys
from copy import deepcopy
from typing import Any

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .common_types import BoundingBox, Polygon4P

__all__ = ["visualize_page", "visualize_kie_page", "draw_boxes"]


def rect_patch(
    geometry: BoundingBox,
    page_dimensions: tuple[int, int],
    label: str | None = None,
    color: tuple[float, float, float] = (0, 0, 0),
    alpha: float = 0.3,
    linewidth: int = 2,
    fill: bool = True,
    preserve_aspect_ratio: bool = False,
) -> patches.Rectangle:
    """Create a matplotlib rectangular patch for the element

    Args:
        geometry: bounding box of the element
        page_dimensions: dimensions of the Page in format (height, width)
        label: label to display when hovered
        color: color to draw box
        alpha: opacity parameter to fill the boxes, 0 = transparent
        linewidth: line width
        fill: whether the patch should be filled
        preserve_aspect_ratio: pass True if you passed True to the predictor

    Returns:
        a rectangular Patch
    """
    if len(geometry) != 2 or any(not isinstance(elt, tuple) or len(elt) != 2 for elt in geometry):
        raise ValueError("invalid geometry format")

    # Unpack
    height, width = page_dimensions
    (xmin, ymin), (xmax, ymax) = geometry
    # Switch to absolute coords
    if preserve_aspect_ratio:
        width = height = max(height, width)
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
    geometry: np.ndarray,
    page_dimensions: tuple[int, int],
    label: str | None = None,
    color: tuple[float, float, float] = (0, 0, 0),
    alpha: float = 0.3,
    linewidth: int = 2,
    fill: bool = True,
    preserve_aspect_ratio: bool = False,
) -> patches.Polygon:
    """Create a matplotlib polygon patch for the element

    Args:
        geometry: bounding box of the element
        page_dimensions: dimensions of the Page in format (height, width)
        label: label to display when hovered
        color: color to draw box
        alpha: opacity parameter to fill the boxes, 0 = transparent
        linewidth: line width
        fill: whether the patch should be filled
        preserve_aspect_ratio: pass True if you passed True to the predictor

    Returns:
        a polygon Patch
    """
    if not geometry.shape == (4, 2):
        raise ValueError("invalid geometry format")

    # Unpack
    height, width = page_dimensions
    geometry[:, 0] = geometry[:, 0] * (max(width, height) if preserve_aspect_ratio else width)
    geometry[:, 1] = geometry[:, 1] * (max(width, height) if preserve_aspect_ratio else height)

    return patches.Polygon(
        geometry,
        fill=fill,
        linewidth=linewidth,
        edgecolor=(*color, alpha),
        facecolor=(*color, alpha),
        label=label,
    )


def create_obj_patch(
    geometry: BoundingBox | Polygon4P | np.ndarray,
    page_dimensions: tuple[int, int],
    **kwargs: Any,
) -> patches.Patch:
    """Create a matplotlib patch for the element

    Args:
        geometry: bounding box (straight or rotated) of the element
        page_dimensions: dimensions of the page in format (height, width)
        **kwargs: keyword arguments for the patch

    Returns:
        a matplotlib Patch
    """
    if isinstance(geometry, tuple):
        if len(geometry) == 2:  # straight word BB (2 pts)
            return rect_patch(geometry, page_dimensions, **kwargs)
        elif len(geometry) == 4:  # rotated word BB (4 pts)
            return polygon_patch(np.asarray(geometry), page_dimensions, **kwargs)
    elif isinstance(geometry, np.ndarray) and geometry.shape == (4, 2):  # rotated line
        return polygon_patch(geometry, page_dimensions, **kwargs)
    raise ValueError("invalid geometry format")


def get_colors(num_colors: int) -> list[tuple[float, float, float]]:
    """Generate num_colors color for matplotlib

    Args:
        num_colors: number of colors to generate

    Returns:
        colors: list of generated colors
    """
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def visualize_page(
    page: dict[str, Any],
    image: np.ndarray,
    words_only: bool = True,
    display_artefacts: bool = True,
    scale: float = 10,
    interactive: bool = True,
    add_labels: bool = True,
    **kwargs: Any,
) -> Figure:
    """Visualize a full page with predicted blocks, lines and words

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
        **kwargs: keyword arguments for the polygon patch

    Returns:
        the matplotlib figure
    """
    # Get proper scale and aspect ratio
    h, w = image.shape[:2]
    size = (scale * w / h, scale) if h > w else (scale, h / w * scale)
    fig, ax = plt.subplots(figsize=size)
    # Display the image
    ax.imshow(image)
    # hide both axis
    ax.axis("off")

    if interactive:
        artists: list[patches.Patch] = []  # instantiate an empty list of patches (to be drawn on the page)

    for block in page["blocks"]:
        if not words_only:
            rect = create_obj_patch(
                block["geometry"], page["dimensions"], label="block", color=(0, 1, 0), linewidth=1, **kwargs
            )
            # add patch on figure
            ax.add_patch(rect)
            if interactive:
                # add patch to cursor's artists
                artists.append(rect)

        for line in block["lines"]:
            if not words_only:
                rect = create_obj_patch(
                    line["geometry"], page["dimensions"], label="line", color=(1, 0, 0), linewidth=1, **kwargs
                )
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)

            for word in line["words"]:
                rect = create_obj_patch(
                    word["geometry"],
                    page["dimensions"],
                    label=f"{word['value']} (confidence: {word['confidence']:.2%})",
                    color=(0, 0, 1),
                    **kwargs,
                )
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)
                elif add_labels:
                    if len(word["geometry"]) == 5:
                        text_loc = (
                            int(page["dimensions"][1] * (word["geometry"][0] - word["geometry"][2] / 2)),
                            int(page["dimensions"][0] * (word["geometry"][1] - word["geometry"][3] / 2)),
                        )
                    else:
                        text_loc = (
                            int(page["dimensions"][1] * word["geometry"][0][0]),
                            int(page["dimensions"][0] * word["geometry"][0][1]),
                        )

                    if len(word["geometry"]) == 2:
                        # We draw only if boxes are in straight format
                        ax.text(
                            *text_loc,
                            word["value"],
                            size=10,
                            alpha=0.5,
                            color=(0, 0, 1),
                        )

        if display_artefacts:
            for artefact in block["artefacts"]:
                rect = create_obj_patch(
                    artefact["geometry"],
                    page["dimensions"],
                    label="artefact",
                    color=(0.5, 0.5, 0.5),
                    linewidth=1,
                    **kwargs,
                )
                ax.add_patch(rect)
                if interactive:
                    artists.append(rect)

    if interactive:
        import mplcursors

        # Create mlp Cursor to hover patches in artists
        mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    fig.tight_layout(pad=0.0)

    return fig


def visualize_kie_page(
    page: dict[str, Any],
    image: np.ndarray,
    words_only: bool = False,
    display_artefacts: bool = True,
    scale: float = 10,
    interactive: bool = True,
    add_labels: bool = True,
    **kwargs: Any,
) -> Figure:
    """Visualize a full page with predicted blocks, lines and words

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from doctr.utils.visualization import visualize_page
    >>> from doctr.models import ocr_db_crnn
    >>> model = ocr_db_crnn(pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([[input_page]])
    >>> visualize_kie_page(out[0].pages[0].export(), input_page)
    >>> plt.show()

    Args:
        page: the exported Page of a Document
        image: np array of the page, needs to have the same shape than page['dimensions']
        words_only: whether only words should be displayed
        display_artefacts: whether artefacts should be displayed
        scale: figsize of the largest windows side
        interactive: whether the plot should be interactive
        add_labels: for static plot, adds text labels on top of bounding box
        **kwargs: keyword arguments for the polygon patch

    Returns:
        the matplotlib figure
    """
    # Get proper scale and aspect ratio
    h, w = image.shape[:2]
    size = (scale * w / h, scale) if h > w else (scale, h / w * scale)
    fig, ax = plt.subplots(figsize=size)
    # Display the image
    ax.imshow(image)
    # hide both axis
    ax.axis("off")

    if interactive:
        artists: list[patches.Patch] = []  # instantiate an empty list of patches (to be drawn on the page)

    colors = {k: color for color, k in zip(get_colors(len(page["predictions"])), page["predictions"])}
    for key, value in page["predictions"].items():
        for prediction in value:
            if not words_only:
                rect = create_obj_patch(
                    prediction["geometry"],
                    page["dimensions"],
                    label=f"{key} \n {prediction['value']} (confidence: {prediction['confidence']:.2%}",
                    color=colors[key],
                    linewidth=1,
                    **kwargs,
                )
                # add patch on figure
                ax.add_patch(rect)
                if interactive:
                    # add patch to cursor's artists
                    artists.append(rect)

    if interactive:
        import mplcursors

        # Create mlp Cursor to hover patches in artists
        mplcursors.Cursor(artists, hover=2).connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    fig.tight_layout(pad=0.0)

    return fig


def draw_boxes(boxes: np.ndarray, image: np.ndarray, color: tuple[int, int, int] | None = None, **kwargs) -> None:
    """Draw an array of relative straight boxes on an image

    Args:
        boxes: array of relative boxes, of shape (*, 4)
        image: np array, float32 or uint8
        color: color to use for bounding box edges
        **kwargs: keyword arguments from `matplotlib.pyplot.plot`
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
            image, (xmin, ymin), (xmax, ymax), color=color if isinstance(color, tuple) else (0, 0, 255), thickness=2
        )
    plt.imshow(image)
    plt.plot(**kwargs)

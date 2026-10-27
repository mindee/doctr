# Rotated documents

Scans are rarely perfectly aligned: pages are skewed, photographed at an angle, fed upside down into the scanner,
or contain vertical text. docTR handles all of these cases, at a small cost in speed. This page helps you choose
the right options.

## Which setting do I need?

| Your documents                                                      | Recommended options                                                |
| ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Upright pages, horizontal text (digital PDFs, clean scans)           | defaults (`assume_straight_pages=True`): fastest                   |
| Slightly skewed or rotated pages, and you want **straight boxes**    | `straighten_pages=True`                                            |
| Pages in any orientation, text in several directions                 | `assume_straight_pages=False`                                      |
| As above, but your application only understands rectangles           | `assume_straight_pages=False, export_as_straight_boxes=True`       |
| You only want to **know** how each page is rotated                   | `detect_orientation=True`                                          |

## Straight pages (default)

With `assume_straight_pages=True`, docTR fits axis-aligned boxes to the text and reads the crops as they are.
This is the fastest mode and the right one for most digital documents. Geometries are
`((xmin, ymin), (xmax, ymax))` pairs.

## Straighten the pages first

`straighten_pages=True` estimates the rotation of each page, rotates the page back, and only then runs text
detection and recognition. The output looks as if the page had been scanned straight: straight boxes, and a
reading order that follows the text.

```python
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True, straighten_pages=True)
```

The coordinates refer to the **straightened** page, which may have different dimensions than the input (check
`page.dimensions`). To get coordinates on the original page instead, add `preserve_original_coords=True`.

## Rotated boxes

`assume_straight_pages=False` makes the detection model predict **rotated boxes**, and enables two small
classifiers:

- a **page orientation** classifier, which estimates the general orientation of each page;
- a **crop orientation** classifier, which rotates each word crop upright (0°, 90°, 180° or 270°) before it is read.

```python
predictor = ocr_predictor(pretrained=True, assume_straight_pages=False)
result = predictor(pages)

word = result.pages[0].blocks[0].lines[0].words[0]
word.geometry  # four (x, y) points: ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
word.crop_orientation  # {'value': 0, 'confidence': 1.0}
```

Add `export_as_straight_boxes=True` to get the enclosing axis-aligned boxes instead of polygons.

## Detect the page orientation

`detect_orientation=True` adds the estimated rotation of each page, in degrees, to the results. It works with every
other option:

```python
predictor = ocr_predictor(pretrained=True, assume_straight_pages=False, detect_orientation=True)
result = predictor(pages)

print(result.pages[0].orientation)  # {'value': -10, 'confidence': None} for a page skewed by 10°
```

## Speed up the orientation models

The orientation classifiers are light, but you can skip them when you know your documents well:

`disable_page_orientation=True`
: Skip the page orientation classifier, e.g. for pages that are only slightly skewed (between about -45° and 45°).
  Only relevant with `assume_straight_pages=False`, `straighten_pages=True` or `detect_orientation=True`.

`disable_crop_orientation=True`
: Skip the word crop classifier when all the text is horizontal (never upside down or vertical). Only relevant
  with `assume_straight_pages=False` or `straighten_pages=True`.

```python
predictor = ocr_predictor(
    pretrained=True,
    assume_straight_pages=False,
    disable_page_orientation=True,
    disable_crop_orientation=True,
)
```

## Learn more

- The [Dealing with rotated documents](https://colab.research.google.com/github/mindee/notebooks/blob/main/doctr/dealing_with_rotations.ipynb)
  notebook compares all these modes on real examples.
- The orientation classifiers can be [retrained on your data](../training/custom_models.md#orientation-classifiers).

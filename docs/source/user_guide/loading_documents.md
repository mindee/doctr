# Loading documents

Predictors work on **pages**: a list of NumPy arrays of shape `(H, W, 3)`, in RGB order and `uint8` type.
{class}`~doctr.io.DocumentFile` reads PDFs, images and web pages into that format, so you rarely have to think
about it.

## From a PDF

```python
from doctr.io import DocumentFile

pages = DocumentFile.from_pdf("path/to/report.pdf")
print(len(pages), pages[0].shape)  # e.g. 3 (2339, 1653, 3)
```

Every page of the PDF is rendered to an image (with [pypdfium2](https://pypdfium2.readthedocs.io)), so
`pages[i]` and `result.pages[i]` always correspond to the *i*-th page of the file.

::::{tab-set}

:::{tab-item} Rendering resolution
The `scale` parameter controls the rendering resolution, where `1` corresponds to 72 dpi. The default, `2`
(144 dpi), is a good trade-off for most documents; increase it for small print:

```python
pages = DocumentFile.from_pdf("path/to/report.pdf", scale=3)  # 216 dpi
```
:::

:::{tab-item} Encrypted PDF
```python
pages = DocumentFile.from_pdf("path/to/protected.pdf", password="s3cr3t")
```
:::

:::{tab-item} From memory
Pass the raw bytes, e.g. a file received by a web API or downloaded from object storage:

```python
with open("path/to/report.pdf", "rb") as f:
    pages = DocumentFile.from_pdf(f.read())
```
:::
::::

:::{tip}
docTR always runs OCR on the rendered pages, even when the PDF already contains a text layer. This gives
consistent results on scanned and digital-born PDFs alike.
:::

## From images

Any format supported by OpenCV works (PNG, JPEG, TIFF, BMP, WebP...). Pass one path, or several to build a
multi-page document:

```python
from doctr.io import DocumentFile

# A single image is a one-page document
pages = DocumentFile.from_images("path/to/scan.jpg")

# Several images are read as consecutive pages of the same document
pages = DocumentFile.from_images(["path/to/page1.png", "path/to/page2.png"])

# Raw bytes work too
with open("path/to/scan.jpg", "rb") as f:
    pages = DocumentFile.from_images(f.read())
```

:::{note}
A multi-frame TIFF is read as a single image (its first frame). Split it into frames first, e.g. with Pillow,
and pass them as NumPy arrays (see below).
:::

## From a web page

Web pages are first printed to PDF with [WeasyPrint](https://weasyprint.org), so this requires the `html` extra
(`pip install "python-doctr[html]"`):

```python
from doctr.io import DocumentFile

pages = DocumentFile.from_url("https://www.example.com")
```

## From NumPy arrays

If your images are already in memory, e.g. frames from a camera or pages produced by another library,
pass them directly to the predictor as a list of arrays:

```python
import numpy as np
from PIL import Image

from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True)

page = np.asarray(Image.open("path/to/scan.png").convert("RGB"))  # (H, W, 3), uint8, RGB
result = predictor([page])
```

:::{warning}
Make sure arrays are **RGB** (OpenCV loads images as BGR: convert them with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`),
of type `uint8`, and have **3 channels** (convert grayscale images with `np.stack([img] * 3, axis=-1)`).
:::

## Mixing sources

`DocumentFile` methods return plain Python lists, so you can concatenate them into a single document:

```python
pages = DocumentFile.from_pdf("cover_letter.pdf") + DocumentFile.from_images(["id_front.jpg", "id_back.jpg"])
result = predictor(pages)  # one Document with all the pages, in order
```

## Low-level readers

`DocumentFile` is a thin wrapper around these functions, which you can use directly when you need more control:

| Function                                      | Reads                                                     |
| --------------------------------------------- | --------------------------------------------------------- |
| {func}`~doctr.io.read_pdf`                       | A PDF into a list of arrays (`scale`, `password`, `rgb_mode`) |
| {func}`~doctr.io.read_img_as_numpy`              | One image into an array (optionally resized)               |
| {func}`~doctr.io.read_img_as_tensor`             | One image into a `torch.Tensor`                            |
| {func}`~doctr.io.decode_img_as_tensor`           | Image bytes into a `torch.Tensor`                          |
| {func}`~doctr.io.read_html`                      | A web page into PDF bytes                                  |

## Next steps

Your pages are ready: head to [](ocr_predictor.md) to configure the predictor that will read them.

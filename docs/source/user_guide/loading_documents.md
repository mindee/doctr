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

{meth}`~doctr.io.DocumentFile.from_files` does the same in a single call, for any mix of PDFs, images, folders and
URLs (folders are expanded to the PDFs and images they contain, sorted by name):

```python
pages = DocumentFile.from_files(["cover_letter.pdf", "id_front.jpg", "id_back.jpg", "path/to/scans/"])
```

## Long documents & large collections

Every page of a list is decoded in memory: at the default resolution (`scale=2`, 1684 x 1191 pixels), an A4 page
weighs about 6 MB, so a 1,000-page PDF would need about 6 GB. Pass `lazy=True` to get a **page source** instead: it behaves like the list (`len`,
indexing, slicing, iteration), but a page is only decoded when it is accessed.

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

predictor = ocr_predictor(pretrained=True)

doc = DocumentFile.from_pdf("path/to/1000_pages.pdf", lazy=True)
print(len(doc), doc[0].shape)  # only the first page is rendered

result = predictor(doc)  # one call, a bounded memory footprint
```

The predictor processes a page source chunk by chunk (`page_chunk_size` pages at a time, 32 by default), and decodes
the next chunk in a background thread while the models run on the current one. The pages of the returned
{class}`~doctr.io.Document` only keep a reference to their image: `result.pages[i].page` renders it again on access,
so `show()` and the figure exports keep working. Pages transformed by the predictor (with `straighten_pages=True`)
cannot be rendered again from the source: as for a list of pages, each of them keeps its straightened image in memory.
Lists of pages already held in memory are simply processed chunk by chunk, without a background thread.

The same goes for images, folders and mixed inputs:

```python
# Every image of a folder (sorted by name), decoded on demand
doc = DocumentFile.from_images("path/to/100k_scans/", lazy=True)

# Any mix of PDFs, images, folders and URLs
doc = DocumentFile.from_files(["contracts/", "annex.pdf", "signature.png"], lazy=True)
source_idx, page_idx = doc.locate(42)  # the input that the 43rd page comes from
```

To write the results as you go instead of keeping them all, iterate over
{meth}`~doctr.models.predictor.OCRPredictor.stream`: it yields the pages one by one, as soon as they are processed.

```python
with open("result.txt", "w") as f:
    for page in predictor.stream(doc):
        f.write(f"--- page {page.page_idx + 1} ---\n{page.render()}\n")
```

| Class                        | Pages                                                              |
| ---------------------------- | ------------------------------------------------------------------ |
| {class}`~doctr.io.PDFSource`    | The pages of a PDF, rendered on demand (`scale`, `password`, `rgb_mode`) |
| {class}`~doctr.io.ImageSource`  | Image files, folders or bytes, decoded on demand                   |
| {class}`~doctr.io.ConcatSource` | The concatenation of several sources (also obtained with `+`)      |

:::{tip}
Missing files, invalid PDFs and files that are not images are reported when the source is created (images are only
checked from their header, a few microseconds per file). Images are decoded when they are processed: one with a
valid header but corrupted content raises a {class}`~doctr.io.PageDecodeError` (a `ValueError`) naming the file.
:::

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

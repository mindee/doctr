# Datasets

{mod}`doctr.datasets` provides ready-to-use public datasets, synthetic data generators, and loaders for your own
annotations. They all return samples in the format expected by the docTR models and training scripts.

## Built-in datasets

Public datasets are downloaded and extracted on first use (`download=True`) into `~/.cache/doctr/datasets`.

```python
from doctr.datasets import CORD

train_set = CORD(train=True, download=True)
sample = train_set[0]
image, target = sample.image, sample.target
```

The same dataset can be loaded for each task:

::::{tab-set}

:::{tab-item} OCR (default)
Full pages, with the boxes **and** the transcriptions of the words:

```python
train_set = CORD(train=True, download=True)
target = train_set[0].target  # {"boxes": array of shape (N, 4), "labels": [str, ...]}
```
:::

:::{tab-item} Detection
Full pages, with the boxes of the words only:

```python
train_set = CORD(train=True, download=True, detection_task=True)
```
:::

:::{tab-item} Recognition
Word crops, with their transcription:

```python
train_set = CORD(train=True, download=True, recognition_task=True)
image, word = train_set[0].image, train_set[0].target
```
:::
::::

Add `use_polygons=True` to get rotated boxes as `(4, 2)` polygons instead of straight boxes (for recognition,
crops are then extracted along the rotated box).

### Available datasets

| Dataset | Detection (train / test pages) | Recognition (train / test crops) | Notes |
| --- | ---: | ---: | --- |
| {class}`~doctr.datasets.FUNSD` | 149 / 50 | 21,888 / 8,707 | English forms |
| {class}`~doctr.datasets.SROIE` | 626 / 360 | 33,608 / 19,342 | Receipts, upper-case labels |
| {class}`~doctr.datasets.CORD` | 800 / 100 | 19,370 / 2,186 | Receipts |
| {class}`~doctr.datasets.IIIT5K` | 2,000 / 3,000 | 2,000 / 3,000 | Scene text |
| {class}`~doctr.datasets.SVT` | 100 / 249 | 257 / 647 | Street view, upper-case labels |
| {class}`~doctr.datasets.SVHN` | 33,402 / 13,068 | 73,257 / 26,032 | House numbers, digits only |
| {class}`~doctr.datasets.SynthText` | 772,875 / 85,875 | ~7,100,000 / 707,470 | Synthetic |
| {class}`~doctr.datasets.IC03` | 246 / 249 | 1,156 / 1,107 | Scene text |
| {class}`~doctr.datasets.IC13` | 229 / 233 | 849 / 1,095 | Scene text, manual download |
| {class}`~doctr.datasets.IMGUR5K` | 7,149 / 796 | 207,901 / 22,672 | Handwritten, manual download |
| {class}`~doctr.datasets.MJSynth` | - | 7,581,382 / 1,337,891 | Synthetic words, manual download |
| {class}`~doctr.datasets.IIITHWS` | - | 7,141,797 / 793,533 | Handwritten words, manual download |
| {class}`~doctr.datasets.WILDRECEIPT` | 1,268 / 472 | 49,377 / 19,598 | Receipts, manual download |
| {class}`~doctr.datasets.COCOTEXT` | 13,880 / 3,261 | 59,130 / 13,258 | Natural images, legible text only, manual download |
| {class}`~doctr.datasets.DocArtefacts` | 2,700 / 300 | - | Object detection: QR codes, bar codes, logos, photos |

Datasets marked *manual download* must be fetched from their authors' website first: pass the paths of the
extracted files to the constructor (see the [API reference](../api/datasets.md)).

## Synthetic data

Generators render random characters or words on the fly, which is useful to pre-train a model or to add an
alphabet to it. Both are used by the reference training scripts when no dataset is given.

```python
from doctr.datasets import VOCABS, CharacterGenerator, WordGenerator

# Single characters, e.g. to train a character classifier
chars = CharacterGenerator(vocab=VOCABS["german"], num_samples=1000)

# Random words of 1 to 12 characters
words = WordGenerator(vocab=VOCABS["german"], min_chars=1, max_chars=12, num_samples=10000)
sample = words[0]  # sample.image: rendered word, sample.target: its text
```

Both accept a `font_family` (one font or a list of fonts) to render with your own fonts.

## Your own data

Load annotated data stored in the [training format](train.md) of each task:

| Class | Task | Expected files |
| --- | --- | --- |
| {class}`~doctr.datasets.DetectionDataset` | Text detection | `images/` + `labels.json` with polygons (optionally per class) |
| {class}`~doctr.datasets.RecognitionDataset` | Text recognition | word images + `labels.json` mapping each image to its text |
| {class}`~doctr.datasets.OCRDataset` | End-to-end OCR | `images/` + `labels.json` with boxes and transcriptions |
| {class}`~doctr.datasets.LayoutDataset` | Layout analysis | `images/` + `labels.json` with polygons and classes |
| {class}`~doctr.datasets.TableStructureDataset` | Table structure | `images/` + `labels.json` with cells and their logical coordinates |
| {class}`~doctr.datasets.OrientationDataset` | Orientation classification | a folder of upright images |

```python
from doctr.datasets import DetectionDataset, OCRDataset, RecognitionDataset

det_set = DetectionDataset(img_folder="path/to/images", label_path="path/to/labels.json")
reco_set = RecognitionDataset(img_folder="path/to/images", labels_path="path/to/labels.json")
ocr_set = OCRDataset(img_folder="path/to/images", label_file="path/to/labels.json")
```

## Transforms and batching

Pass transformations with `img_transforms` (applied to the image) and `sample_transforms` (applied to the image
and its target, e.g. rotations). Datasets return images of different sizes, so resize them before batching. Batch
with a PyTorch {class}`~torch.utils.data.DataLoader` and the `collate_fn` of the dataset:

```python
from torch.utils.data import DataLoader

from doctr import transforms as T
from doctr.datasets import RecognitionDataset

train_set = RecognitionDataset(
    img_folder="path/to/images",
    labels_path="path/to/labels.json",
    img_transforms=T.Resize((32, 128), preserve_aspect_ratio=True),
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=train_set.collate_fn)
images, targets = next(iter(train_loader))  # images: (32, 3, 32, 128) tensor, targets: list of 32 strings
```

{mod}`doctr.transforms` provides document-specific augmentations (shadows, glare, perspective, rotations...). See
the [API reference](../api/transforms.md) and the reference training scripts for complete pipelines.

# Formula recognition

Formula recognition models read the image of a mathematical formula and predict its LaTeX source. In the OCR
pipeline, they read the `Formula` regions found by the [layout model](layout.md), which are exported as display math
(e.g. `$$...$$` in Markdown).

## Architectures

| Architecture | Paper |
| --- | --- |
| {func}`~doctr.models.formula.formulanet` {bdg-secondary}`default` | [PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition](https://arxiv.org/abs/2503.18382) |

`formulanet` is made of a convolutional encoder, {func}`~doctr.models.classification.hgnetv2_b4`, and a small
autoregressive transformer decoder:

- the encoder turns the formula image into a feature map (stride 32, 2048 channels), flattened into a sequence and
  projected to 384 dimensions, with 2D positional encodings;
- the decoder (4 layers, 16 heads, 1536 hidden units) predicts the LaTeX tokens one by one, greedily, with cached
  keys and values. It learns formulas of up to 4096 tokens (`max_length`) and generates up to 2048 tokens per formula
  (`max_decode_length`).

The decoding of a batch stops once all its formulas are complete, so its time follows its longest formula.
`max_decode_length` caps the time spent on a formula the model fails to end (e.g. a repetition loop), which would hold
up its whole batch: set it with `formula_predictor(..., max_decode_length=...)`.

| Architecture | Input shape | # params | Encoder | Decoder |
| --- | --- | ---: | ---: | ---: |
| `formulanet` | (256, 768, 3) | 26.4 M | 13.6 M | 12.9 M |

The input is wide rather than square, so long equations are not shrunk: each formula is resized to fit the input while
keeping its aspect ratio, and padded. The model is trained with random margins around the formulas, so the crops of
layout regions are read as they are, margins included.

The pretrained weights are trained on the 1.06 M printed and handwritten formulas of
[UniMER-1M](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) (with the train split of
[HME100K](https://huggingface.co/datasets/ConstantHao/HME100K), which UniMER-1M lists without images), starting from the
pretrained `hgnetv2_b4` backbone. Fine-tune them on your data with the
[reference scripts](https://github.com/mindee/doctr/tree/main/references/formula#readme).

## LaTeX tokens

The model predicts LaTeX **tokens** from a fixed list of 1,324 tokens, {data}`~doctr.datasets.LATEX_TOKENS` (there is
no BPE): the printable ASCII characters, the control symbols (`\{`, `\,`, `\\`...), the commands (`\frac`, `\alpha`,
`\operatorname*`...) and environments (`\begin{array}`...) supported by [KaTeX](https://katex.org/docs/supported), and
every other command and character of the [UniMER](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) and
[HME100K](https://huggingface.co/datasets/ConstantHao/HME100K) labels (e.g. `\ensuremath`, `\textsc`, Greek, Cyrillic and
CJK characters), so these datasets are fully covered. The prediction is the tokens separated by spaces, e.g.
`\frac { a } { b ^ { 2 } }`, which renders the same as `\frac{a}{b^{2}}`.

{func}`~doctr.datasets.tokenize_latex` and {func}`~doctr.datasets.normalize_latex` split any LaTeX string into the
same tokens, so labels can be written in the usual way:

```python
from doctr.datasets import normalize_latex

normalize_latex(r"\sum_{i=1}^{n} x_i")  # '\\sum _ { i = 1 } ^ { n } x _ i'
```

A custom list of tokens (e.g. with more CJK characters, for the formulas of Chinese documents) is passed with
`formulanet(vocab=...)`.

:::{note}
About 1.3% of the UniMER-1M labels use TeX commands that KaTeX does not render (e.g. `\ensuremath`, `\Tilde`,
`\boldmath`, or `\eqno` for an equation number), and the model can predict them. Such a formula renders in LaTeX, but
not in a KaTeX-based Markdown or HTML viewer.
:::

## Benchmark

The metrics are produced by `references/formula/evaluate.py` with {class}`~doctr.utils.metrics.FormulaMetric`, on the
LaTeX tokens: the **exact match** rate, the **edit similarity** (one minus the relative token edit distance) and the
corpus-level **BLEU-4** score.

Results of the pretrained `formulanet` on [UniMER-Test](https://huggingface.co/datasets/wanderkid/UniMER_Dataset)
(23,757 formulas), with greedy decoding:

| Subset | Formulas | Median tokens | Exact match | Edit similarity | BLEU |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPE (simple printed) | 6,762 | 48 | 45.92 | 93.86 | 0.833 |
| CPE (complex printed) | 5,921 | 192 | 21.53 | 92.39 | 0.804 |
| SCE (screen-captured) | 4,742 | 9 | 36.80 | 80.10 | 0.431 |
| HWE (handwritten) | 6,332 | 16 | 61.96 | 93.61 | 0.887 |
| **All** | 23,757 | 32 | 44.27 | 91.26 | 0.838 |

Exact match requires every token to be right, so it drops for long formulas (CPE) while the edit similarity stays
high. BLEU-4 is harsh on very short formulas (SCE): a single wrong token removes most of their 4-grams. It also counts different spellings of the same formula (e.g. `\left(` vs `(`, extra braces or spacing) as
errors: compare models on the edit similarity and BLEU, per subset.

## Usage

```python
from doctr.models import formula_predictor, ocr_predictor

# In the OCR pipeline: formulas are located by the layout model, then read as LaTeX
predictor = ocr_predictor(pretrained=True, detect_formulas=True)

# On its own, on images of formulas
formulas = formula_predictor("formulanet", pretrained=True)
```

See [](../user_guide/layout_and_tables.md#formula-recognition) to work with the recognized formulas.

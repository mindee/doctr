# Formula recognition

Formula recognition models read the image of a mathematical formula and predict its LaTeX source. In the OCR
pipeline, they read the `Formula` regions found by the [layout model](layout.md), which are exported as display math
(e.g. `$$...$$` in Markdown).

## Architectures

| Architecture | Paper | Pretrained weights |
| --- | --- | --- |
| {func}`~doctr.models.formula.formulanet` {bdg-secondary}`default` | [PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition](https://arxiv.org/abs/2503.18382) | {bdg-success}`available` |

`formulanet` combines a {func}`~doctr.models.classification.hgnetv2_b4` encoder with a small transformer decoder,
which predicts LaTeX tokens from {data}`~doctr.datasets.LATEX_TOKENS`, e.g. `\frac { a } { b ^ { 2 } }`. The
pretrained model is trained on the printed and handwritten formulas of
[UniMER-1M](https://huggingface.co/datasets/wanderkid/UniMER_Dataset).

## Benchmark

| Architecture | Input shape | # params | Exact match | Edit similarity | BLEU |
| --- | --- | ---: | ---: | ---: | ---: |
| `formulanet` | (256, 768, 3) | 26.4 M | 44.40 | 91.30 | 0.840 |

The metrics are produced by `references/formula/evaluate.py` with {class}`~doctr.utils.metrics.FormulaMetric` on
[UniMER-Test](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) (23,757 formulas), on the LaTeX tokens:
**exact match** rate, **edit similarity** (one minus the relative token edit distance) and corpus-level **BLEU-4**.

:::{note}
Equation numbers are predicted with `\tag{...}` or as text after the formula. A few formulas may still use TeX commands that KaTeX does not render
(e.g. `\boldmath`): they render in LaTeX, but not in a KaTeX-based Markdown or HTML viewer.
:::

## Usage

```python
from doctr.models import formula_predictor, ocr_predictor

# In the OCR pipeline
predictor = ocr_predictor(pretrained=True, detect_formulas=True)

# On its own
formulas = formula_predictor("formulanet", pretrained=True)
```

See [](../user_guide/layout_and_tables.md#formula-recognition) to work with the recognized formulas, and the
[reference scripts](https://github.com/mindee/doctr/tree/main/references/formula#readme) to train on your own data.

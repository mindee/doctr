# Table structure recognition

Table structure models parse the image of a table into a machine-understandable grid: they localize every cell
(spatial structure) and recover the rows and columns it spans (logical structure).

## Architectures

| Architecture | Paper |
| --- | --- |
| {func}`~doctr.models.table_structure.tablecenternet` {bdg-secondary}`default` | [TableCenterNet: A one-stage network for table structure recognition](https://arxiv.org/abs/2504.17522) |

## Benchmark

| Architecture | Input shape | # params | Recall | Precision | F1 | Structure acc. | sec/it (batch 1) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tablecenternet` | (1024, 1024, 3) | 7.1 M | 82.31 | 96.01 | 88.64 | 77.53 | 0.5 |

The metrics are produced by `references/table/evaluate.py` with
{class}`~doctr.utils.metrics.TableCellMetric`: cell detection **recall**, **precision** and **F1** (cells matched
above an IoU of 0.5), and **structure accuracy**, the share of matched cells whose row and column spans are correct.

## Usage

```python
from doctr.models import ocr_predictor, table_predictor

# In the OCR pipeline: tables are located by the layout model, then parsed
predictor = ocr_predictor(pretrained=True, detect_tables=True)

# On its own, on images of tables
tables = table_predictor("tablecenternet", pretrained=True)
```

See [](../user_guide/layout_and_tables.md#table-structure-recognition) to work with the recognized tables.

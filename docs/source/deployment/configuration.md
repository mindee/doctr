# Configuration

docTR has no configuration file: predictors are configured in code, and a few environment variables control how
the library interacts with the system.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `DOCTR_CACHE_DIR` | `~/.cache/doctr` | Where downloaded model weights (`models/`) and datasets (`datasets/`) are stored |
| `DOCTR_MULTIPROCESSING_DISABLE` | unset | Set to `TRUE` to disable the thread pools used for data processing, e.g. where `/dev/shm` is not writable |

## Model cache

Pretrained weights are downloaded the first time a model is used, then loaded from the cache. Their integrity is
checked with the hash in the file name.

```bash
export DOCTR_CACHE_DIR=/opt/models/doctr
```

### Offline use

To run on a machine without internet access, populate the cache on a connected machine, then copy it:

```python
# On a connected machine: instantiating the predictors downloads their weights
from doctr.models import ocr_predictor

ocr_predictor(pretrained=True)  # detection + recognition
ocr_predictor(pretrained=True, assume_straight_pages=False)  # + orientation classifiers
ocr_predictor(pretrained=True, detect_tables=True)  # + layout and table models
```

Copy the cache directory (`~/.cache/doctr` or `$DOCTR_CACHE_DIR`) to the same location on the offline machine.
Alternatively, load the weight files from any path with `model.from_pretrained("path/to/weights.pt")` (see
[](../training/custom_models.md)).

## AWS Lambda

The [AWS Lambda](https://aws.amazon.com/lambda/) runtime only allows writing to `/tmp`, and does not provide the
shared memory used by multiprocessing. Configure the function with:

```bash
DOCTR_MULTIPROCESSING_DISABLE=TRUE
DOCTR_CACHE_DIR=/tmp/doctr
```

:::{tip}
Downloading the weights at every cold start is slow. Package them in your container image or a Lambda layer, and
point `DOCTR_CACHE_DIR` to their location if it is writable, or load them with `from_pretrained`.
:::

The same settings apply to other restricted environments (read-only containers, some serverless platforms).

## Logging

docTR logs through the standard {mod}`logging` module, under the `doctr` logger:

```python
import logging

logging.getLogger("doctr").setLevel(logging.WARNING)
```

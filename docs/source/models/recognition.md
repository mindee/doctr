# Text recognition

Text recognition models transcribe the characters of a word crop. They output a string and a confidence score, and
can only predict the characters of the [vocabulary](vocabs.md) they were trained on.

## Architectures

| Architecture | Paper | Decoding |
| --- | --- | --- |
| {func}`~doctr.models.recognition.crnn_vgg16_bn` {bdg-secondary}`default`, {func}`~doctr.models.recognition.crnn_mobilenet_v3_small`, {func}`~doctr.models.recognition.crnn_mobilenet_v3_large` | [An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/pdf/1507.05717.pdf) | CTC |
| {func}`~doctr.models.recognition.sar_resnet31` | [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/pdf/1811.00751.pdf) | Attention |
| {func}`~doctr.models.recognition.master` | [MASTER: Multi-Aspect Non-local Network for Scene Text Recognition](https://arxiv.org/pdf/1910.02562.pdf) | Transformer |
| {func}`~doctr.models.recognition.vitstr_small`, {func}`~doctr.models.recognition.vitstr_base` | [Vision Transformer for Fast and Efficient Scene Text Recognition](https://arxiv.org/pdf/2105.08582.pdf) | Parallel |
| {func}`~doctr.models.recognition.parseq` | [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/pdf/2207.06966) | Permuted autoregressive |
| {func}`~doctr.models.recognition.viptr_tiny` | [A Vision Permutable Extractor for Fast and Efficient Scene Text Recognition](https://arxiv.org/abs/2401.10110) | CTC |

## Benchmark

Word accuracy on FUNSD and CORD word crops (exact and partial match, see {class}`~doctr.utils.metrics.TextMatch`), and latency with a batch size of 64:

| Architecture | Input shape | # params | FUNSD exact | FUNSD partial | CORD exact | CORD partial | sec/it |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `crnn_vgg16_bn` | (32, 128, 3) | 15.8 M | 88.21 | 88.95 | 95.47 | 95.91 | 0.6 |
| `crnn_mobilenet_v3_small` | (32, 128, 3) | 2.1 M | 87.25 | 87.99 | 93.91 | 94.34 | **0.05** |
| `crnn_mobilenet_v3_large` | (32, 128, 3) | 4.5 M | 87.38 | 88.09 | 94.46 | 94.92 | 0.08 |
| `master` | (32, 128, 3) | 58.7 M | **88.57** | **89.39** | **95.73** | **96.21** | 17.6 |
| `sar_resnet31` | (32, 128, 3) | 55.4 M | 88.10 | 88.88 | 94.83 | 95.29 | 4.9 |
| `vitstr_small` | (32, 128, 3) | 21.4 M | 88.00 | 88.82 | 95.40 | 95.78 | 1.5 |
| `vitstr_base` | (32, 128, 3) | 85.2 M | 88.33 | 89.09 | 95.32 | 95.71 | 4.1 |
| `parseq` | (32, 128, 3) | 23.8 M | 88.53 | 89.24 | 95.56 | 95.91 | 2.2 |
| `viptr_tiny` | (32, 128, 3) | 3.2 M | 86.03 | 86.71 | 93.08 | 93.47 | 0.08 |

## Usage

```python
from doctr.models import ocr_predictor, recognition_predictor

# In the OCR pipeline
predictor = ocr_predictor(reco_arch="parseq", pretrained=True)

# On its own, on word crops
reco_predictor = recognition_predictor("parseq", pretrained=True)

# The characters the model can predict
print(reco_predictor.model.cfg["vocab"])
```

## Languages

The pretrained recognition models are trained on the `french` vocabulary: Latin characters with the accents of
the main Western European languages, digits, punctuation and common currency symbols. For other languages:

- restrict the output to the characters you expect with a [whitelist](../user_guide/ocr_predictor.md#restrict-the-output-characters);
- look for a [community model](../training/sharing_models.md#community-models), such as the multilingual PARSeq;
- [train your own](../training/train.md) on one of the [100+ vocabularies](vocabs.md).

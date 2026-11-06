# End-to-end OCR

An OCR predictor combines a detection model with a recognition model: any pair of the architectures listed in this
section works. A word counts as correct when it is both well localized and correctly transcribed.

## Benchmark

| Architecture | FUNSD recall | FUNSD precision | CORD recall | CORD precision |
| --- | ---: | ---: | ---: | ---: |
| `db_resnet50` + `crnn_vgg16_bn` | 73.37 | 76.11 | 84.80 | 79.09 |
| `db_resnet50` + `crnn_mobilenet_v3_small` | 73.06 | 75.79 | 84.64 | 78.94 |
| `db_resnet50` + `crnn_mobilenet_v3_large` | 73.17 | 75.90 | 84.96 | 79.25 |
| `db_resnet50` + `master` | **73.90** | **76.66** | 85.84 | 80.07 |
| `db_resnet50` + `sar_resnet31` | 73.58 | 76.33 | 85.64 | 79.88 |
| `db_resnet50` + `vitstr_small` | 73.06 | 75.79 | **85.95** | **80.17** |
| `db_resnet50` + `vitstr_base` | 73.70 | 76.46 | 85.76 | 79.99 |
| `db_resnet50` + `parseq` | 73.52 | 76.27 | 85.91 | 80.13 |
| *Google Vision - text detection* | 59.50 | 62.50 | 75.30 | 59.03 |
| *Google Vision - document text detection* | 64.00 | 53.30 | 68.90 | 61.10 |
| *AWS Textract* | 78.10 | 83.00 | 87.50 | 66.00 |
| *Azure Form Recognizer (v3.2)* | 79.42 | 85.89 | 89.62 | 88.93 |

Commercial services are shown in italics, for reference.

## Usage

```python
from doctr.models import ocr_predictor

predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="master", pretrained=True)
```

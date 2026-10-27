# Sharing models

docTR is integrated with the [Hugging Face Hub](https://huggingface.co/models?library=doctr): load models shared by
the community in one line, and publish your own.

## Load a model from the Hub

{func}`~doctr.models.factory.from_hub` downloads a model and rebuilds it with its configuration (architecture, vocabulary,
classes...). Plug it into any predictor:

```python
from doctr.io import DocumentFile
from doctr.models import from_hub, ocr_predictor

det_model = from_hub("Felix92/doctr-torch-db-mobilenet-v3-large")
reco_model = from_hub("Felix92/doctr-torch-parseq-multilingual-v1")

predictor = ocr_predictor(det_arch=det_model, reco_arch=reco_model)
result = predictor(DocumentFile.from_images("path/to/page.jpg"))
```

## Publish a model

You need a [Hugging Face account](https://huggingface.co/join) and [Git LFS](https://git-lfs.com) installed.

```python
from doctr.models import login_to_hub, push_to_hf_hub, recognition

login_to_hub()  # paste an access token with write permissions

model = recognition.crnn_mobilenet_v3_large(pretrained=True)
push_to_hf_hub(
    model,
    model_name="doctr-crnn-mobilenet-v3-large-french-v1",
    task="recognition",  # classification, detection, recognition, layout or table_structure
    arch="crnn_mobilenet_v3_large",
)
```

Existing repositories are never overwritten. The reference training scripts can also publish the model at the end
of the training:

```bash
python references/recognition/train.py crnn_mobilenet_v3_large --name doctr-crnn-mobilenet-v3-large --push-to-hub
```

### Naming conventions

| Task | Suggested name |
| --- | --- |
| Classification | `doctr-<architecture>-<vocab>` |
| Detection | `doctr-<architecture>` |
| Recognition | `doctr-<architecture>-<vocab>` |
| Layout | `doctr-<architecture>` |
| Table structure | `doctr-<architecture>` |

## Community models

Models trained and shared by the community. Open a pull request to add yours!

| Task | Architecture | Repository | Language |
| --- | --- | --- | --- |
| Detection | `db_resnet50` | [`rania-sr/doctr-Detection-model-v1-arabic`](https://huggingface.co/rania-sr/doctr-Detection-model-v1-arabic) | Arabic |
| Recognition | `parseq` | [`Felix92/doctr-torch-parseq-multilingual-v1`](https://huggingface.co/Felix92/doctr-torch-parseq-multilingual-v1) | Multilingual |
| Recognition | `parseq` | [`rania-sr/doctr-model-v1-arabic`](https://huggingface.co/rania-sr/doctr-model-v1-arabic) | Arabic |
| Recognition | `crnn_vgg16_bn` | [`tilman-rassy/doctr-crnn-vgg16-bn-fascan-v1`](https://huggingface.co/tilman-rassy/doctr-crnn-vgg16-bn-fascan-v1) | French + German + § |

Dummy models, handy for tests: [`Felix92/doctr-dummy-torch-resnet18`](https://huggingface.co/Felix92/doctr-dummy-torch-resnet18) (classification),
[`Felix92/doctr-dummy-torch-lw-detr-s`](https://huggingface.co/Felix92/doctr-dummy-torch-lw-detr-s) (layout) and
[`Felix92/doctr-dummy-torch-tablecenternet`](https://huggingface.co/Felix92/doctr-dummy-torch-tablecenternet) (table structure).

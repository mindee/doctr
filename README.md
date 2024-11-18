<p align="center">
  <img src="https://github.com/mindee/doctr/raw/main/docs/images/Logo_doctr.gif" width="40%">
</p>

[![Slack Icon](https://img.shields.io/badge/Slack-Community-4A154B?style=flat-square&logo=slack&logoColor=white)](https://slack.mindee.com) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) ![Build Status](https://github.com/mindee/doctr/workflows/builds/badge.svg) [![Docker Images](https://img.shields.io/badge/Docker-4287f5?style=flat&logo=docker&logoColor=white)](https://github.com/mindee/doctr/pkgs/container/doctr) [![codecov](https://codecov.io/gh/mindee/doctr/branch/main/graph/badge.svg?token=577MO567NM)](https://codecov.io/gh/mindee/doctr) [![CodeFactor](https://www.codefactor.io/repository/github/mindee/doctr/badge?s=bae07db86bb079ce9d6542315b8c6e70fa708a7e)](https://www.codefactor.io/repository/github/mindee/doctr) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/340a76749b634586a498e1c0ab998f08)](https://app.codacy.com/gh/mindee/doctr?utm_source=github.com&utm_medium=referral&utm_content=mindee/doctr&utm_campaign=Badge_Grade) [![Doc Status](https://github.com/mindee/doctr/workflows/doc-status/badge.svg)](https://mindee.github.io/doctr) [![Pypi](https://img.shields.io/badge/pypi-v0.10.0-blue.svg)](https://pypi.org/project/python-doctr/) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mindee/doctr) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mindee/notebooks/blob/main/doctr/quicktour.ipynb) [![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20docTR%20Guru-006BFF)](https://gurubase.io/g/doctr)


**Optical Character Recognition made seamless & accessible to anyone, powered by TensorFlow 2 & PyTorch**

What you can expect from this repository:

- efficient ways to parse textual information (localize and identify each word) from your documents
- guidance on how to integrate this in your current architecture

![OCR_example](https://github.com/mindee/doctr/raw/main/docs/images/ocr.png)

## Quick Tour

### Getting your pretrained model

End-to-End OCR is achieved in docTR using a two-stage approach: text detection (localizing words), then text recognition (identify all characters in the word).
As such, you can select the architecture used for [text detection](https://mindee.github.io/doctr/latest/modules/models.html#doctr-models-detection), and the one for [text recognition](https://mindee.github.io/doctr/latest//modules/models.html#doctr-models-recognition) from the list of available implementations.

```python
from doctr.models import ocr_predictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
```

### Reading files

Documents can be interpreted from PDF or images:

```python
from doctr.io import DocumentFile
# PDF
pdf_doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Image
single_img_doc = DocumentFile.from_images("path/to/your/img.jpg")
# Webpage (requires `weasyprint` to be installed)
webpage_doc = DocumentFile.from_url("https://www.yoursite.com")
# Multiple page images
multi_img_doc = DocumentFile.from_images(["path/to/page1.jpg", "path/to/page2.jpg"])
```

### Putting it together

Let's use the default pretrained model for an example:

```python
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
# PDF
doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Analyze
result = model(doc)
```

### Dealing with rotated documents

Should you use docTR on documents that include rotated pages, or pages with multiple box orientations,
you have multiple options to handle it:

- If you only use straight document pages with straight words (horizontal, same reading direction),
consider passing `assume_straight_boxes=True` to the ocr_predictor. It will directly fit straight boxes
on your page and return straight boxes, which makes it the fastest option.

- If you want the predictor to output straight boxes (no matter the orientation of your pages, the final localizations
will be converted to straight boxes), you need to pass `export_as_straight_boxes=True` in the predictor. Otherwise, if `assume_straight_pages=False`, it will return rotated bounding boxes (potentially with an angle of 0°).

If both options are set to False, the predictor will always fit and return rotated boxes.

To interpret your model's predictions, you can visualize them interactively as follows:

```python
# Display the result (requires matplotlib & mplcursors to be installed)
result.show()
```

![Visualization sample](https://github.com/mindee/doctr/raw/main/docs/images/doctr_example_script.gif)

Or even rebuild the original document from its predictions:

```python
import matplotlib.pyplot as plt

synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()
```

![Synthesis sample](https://github.com/mindee/doctr/raw/main/docs/images/synthesized_sample.png)

The `ocr_predictor` returns a `Document` object with a nested structure (with `Page`, `Block`, `Line`, `Word`, `Artefact`).
To get a better understanding of our document model, check our [documentation](https://mindee.github.io/doctr/modules/io.html#document-structure):

You can also export them as a nested dict, more appropriate for JSON format:

```python
json_output = result.export()
```

### Use the KIE predictor

The KIE predictor is a more flexible predictor compared to OCR as your detection model can detect multiple classes in a document. For example, you can have a detection model to detect just dates and addresses in a document.

The KIE predictor makes it possible to use detector with multiple classes with a recognition model and to have the whole pipeline already setup for you.

```python
from doctr.io import DocumentFile
from doctr.models import kie_predictor

# Model
model = kie_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
# PDF
doc = DocumentFile.from_pdf("path/to/your/doc.pdf")
# Analyze
result = model(doc)

predictions = result.pages[0].predictions
for class_name in predictions.keys():
    list_predictions = predictions[class_name]
    for prediction in list_predictions:
        print(f"Prediction for {class_name}: {prediction}")
```

The KIE predictor results per page are in a dictionary format with each key representing a class name and it's value are the predictions for that class.

### If you are looking for support from the Mindee team

[![Bad OCR test detection image asking the developer if they need help](https://github.com/mindee/doctr/raw/main/docs/images/doctr-need-help.png)](https://mindee.com/product/doctr)

## Installation

### Prerequisites

Python 3.10 (or higher) and [pip](https://pip.pypa.io/en/stable/) are required to install docTR.

### Latest release

You can then install the latest release of the package using [pypi](https://pypi.org/project/python-doctr/) as follows:

```shell
pip install python-doctr
```

> :warning: Please note that the basic installation is not standalone, as it does not provide a deep learning framework, which is required for the package to run.

We try to keep framework-specific dependencies to a minimum. You can install framework-specific builds as follows:

```shell
# for TensorFlow
pip install "python-doctr[tf]"
# for PyTorch
pip install "python-doctr[torch]"
# optional dependencies for visualization, html, and contrib modules can be installed as follows:
pip install "python-doctr[torch,viz,html,contib]"
```

For MacBooks with M1 chip, you will need some additional packages or specific versions:

- TensorFlow 2: [metal plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- PyTorch: [version >= 2.0.0](https://pytorch.org/get-started/locally/#start-locally)

### Developer mode

Alternatively, you can install it from source, which will require you to install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
First clone the project repository:

```shell
git clone https://github.com/mindee/doctr.git
pip install -e doctr/.
```

Again, if you prefer to avoid the risk of missing dependencies, you can install the TensorFlow or the PyTorch build:

```shell
# for TensorFlow
pip install -e doctr/.[tf]
# for PyTorch
pip install -e doctr/.[torch]
```

## Models architectures

Credits where it's due: this repository is implementing, among others, architectures from published research papers.

### Text Detection

- DBNet: [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf).
- LinkNet: [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/pdf/1707.03718.pdf)
- FAST: [FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation](https://arxiv.org/pdf/2111.02394.pdf)

### Text Recognition

- CRNN: [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/pdf/1507.05717.pdf).
- SAR: [Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/pdf/1811.00751.pdf).
- MASTER: [MASTER: Multi-Aspect Non-local Network for Scene Text Recognition](https://arxiv.org/pdf/1910.02562.pdf).
- ViTSTR: [Vision Transformer for Fast and Efficient Scene Text Recognition](https://arxiv.org/pdf/2105.08582.pdf).
- PARSeq: [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/pdf/2207.06966).

## More goodies

### Documentation

The full package documentation is available [here](https://mindee.github.io/doctr/) for detailed specifications.

### Demo app

A minimal demo app is provided for you to play with our end-to-end OCR models!

![Demo app](https://github.com/mindee/doctr/raw/main/docs/images/demo_update.png)

#### Live demo

Courtesy of :hugs: [Hugging Face](https://huggingface.co/) :hugs:, docTR has now a fully deployed version available on [Spaces](https://huggingface.co/spaces)!
Check it out [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mindee/doctr)

#### Running it locally

If you prefer to use it locally, there is an extra dependency ([Streamlit](https://streamlit.io/)) that is required.

##### Tensorflow version

```shell
pip install -r demo/tf-requirements.txt
```

Then run your app in your default browser with:

```shell
USE_TF=1 streamlit run demo/app.py
```

##### PyTorch version

```shell
pip install -r demo/pt-requirements.txt
```

Then run your app in your default browser with:

```shell
USE_TORCH=1 streamlit run demo/app.py
```

#### TensorFlow.js

Instead of having your demo actually running Python, you would prefer to run everything in your web browser?
Check out our [TensorFlow.js demo](https://github.com/mindee/doctr-tfjs-demo) to get started!

![TFJS demo](https://github.com/mindee/doctr/raw/main/docs/images/demo_illustration_mini.png)

### Docker container

We offer Docker container support for easy testing and deployment. [Here are the available docker tags.](https://github.com/mindee/doctr/pkgs/container/doctr).

#### Using GPU with docTR Docker Images

The docTR Docker images are GPU-ready and based on CUDA `12.2`. Make sure your host is **at least `12.2`**, otherwise Torch or TensorFlow won't be able to initialize the GPU.
Please ensure that Docker is configured to use your GPU.

To verify and configure GPU support for Docker, please follow the instructions provided in the [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Once Docker is configured to use GPUs, you can run docTR Docker containers with GPU support:

```shell
docker run -it --gpus all ghcr.io/mindee/doctr:torch-py3.9.18-2024-10 bash
```

#### Available Tags

The Docker images for docTR follow a specific tag nomenclature: `<deps>-py<python_version>-<doctr_version|YYYY-MM>`. Here's a breakdown of the tag structure:

- `<deps>`: `tf`, `torch`, `tf-viz-html-contrib` or `torch-viz-html-contrib`.
- `<python_version>`: `3.9.18`, `3.10.13` or `3.11.8`.
- `<doctr_version>`: a tag >= `v0.11.0`
- `<YYYY-MM>`: e.g. `2014-10`

Here are examples of different image tags:

| Tag                        | Description                                       |
|----------------------------|---------------------------------------------------|
| `tf-py3.10.13-v0.11.0`       | TensorFlow version `3.10.13` with docTR `v0.11.0`. |
| `torch-viz-html-contrib-py3.11.8-2024-10`       | Torch with extra dependencies version `3.11.8` from latest commit on `main` in `2024-10`. |
| `torch-py3.11.8-2024-10`| PyTorch version `3.11.8` from latest commit on `main` in `2024-10`. |

#### Building Docker Images Locally

You can also build docTR Docker images locally on your computer.

```shell
docker build -t doctr .
```

You can specify custom Python versions and docTR versions using build arguments. For example, to build a docTR image with TensorFlow, Python version `3.9.10`, and docTR version `v0.7.0`, run the following command:

```shell
docker build -t doctr --build-arg FRAMEWORK=tf --build-arg PYTHON_VERSION=3.9.10 --build-arg DOCTR_VERSION=v0.7.0 .
```

### Example script

An example script is provided for a simple documentation analysis of a PDF or image file:

```shell
python scripts/analyze.py path/to/your/doc.pdf
```

All script arguments can be checked using `python scripts/analyze.py --help`

### Minimal API integration

Looking to integrate docTR into your API? Here is a template to get you started with a fully working API using the wonderful [FastAPI](https://github.com/tiangolo/fastapi) framework.

#### Deploy your API locally

Specific dependencies are required to run the API template, which you can install as follows:

```shell
cd api/
pip install poetry
make lock
pip install -r requirements.txt
```

You can now run your API locally:

```shell
uvicorn --reload --workers 1 --host 0.0.0.0 --port=8002 --app-dir api/ app.main:app
```

Alternatively, you can run the same server on a docker container if you prefer using:

```shell
PORT=8002 docker-compose up -d --build
```

#### What you have deployed

Your API should now be running locally on your port 8002. Access your automatically-built documentation at [http://localhost:8002/redoc](http://localhost:8002/redoc) and enjoy your three functional routes ("/detection", "/recognition", "/ocr", "/kie"). Here is an example with Python to send a request to the OCR route:

```python
import requests

params = {"det_arch": "db_resnet50", "reco_arch": "crnn_vgg16_bn"}

with open('/path/to/your/doc.jpg', 'rb') as f:
    files = [  # application/pdf, image/jpeg, image/png supported
        ("files", ("doc.jpg", f.read(), "image/jpeg")),
    ]
print(requests.post("http://localhost:8080/ocr", params=params, files=files).json())
```

### Example notebooks

Looking for more illustrations of docTR features? You might want to check the [Jupyter notebooks](https://github.com/mindee/doctr/tree/main/notebooks) designed to give you a broader overview.

## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{doctr2021,
    title={docTR: Document Text Recognition},
    author={Mindee},
    year={2021},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/mindee/doctr}}
}
```

## Contributing

If you scrolled down to this section, you most likely appreciate open source. Do you feel like extending the range of our supported characters? Or perhaps submitting a paper implementation? Or contributing in any other way?

You're in luck, we compiled a short guide (cf. [`CONTRIBUTING`](https://mindee.github.io/doctr/contributing/contributing.html)) for you to easily do so!

## License

Distributed under the Apache 2.0 License. See [`LICENSE`](https://github.com/mindee/doctr?tab=Apache-2.0-1-ov-file#readme) for more information.

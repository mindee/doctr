
# DocTR: Document Text Recognition

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) ![Build Status](https://github.com/mindee/doctr/workflows/python-package/badge.svg) [![codecov](https://codecov.io/gh/mindee/doctr/branch/main/graph/badge.svg?token=577MO567NM)](https://codecov.io/gh/mindee/doctr) [![CodeFactor](https://www.codefactor.io/repository/github/mindee/doctr/badge?s=bae07db86bb079ce9d6542315b8c6e70fa708a7e)](https://www.codefactor.io/repository/github/mindee/doctr) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/340a76749b634586a498e1c0ab998f08)](https://app.codacy.com/gh/mindee/doctr?utm_source=github.com&utm_medium=referral&utm_content=mindee/doctr&utm_campaign=Badge_Grade) [![Doc Status](https://github.com/mindee/doctr/workflows/doc-status/badge.svg)](https://mindee.github.io/doctr) [![Pypi](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/python-doctr/) 

Extract valuable information from your documents.



## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [Python package](#python-package)
  * [Docker container](#docker-container)
  * [Example script](#example-script)
  * [Demo app](#demo-app)
* [Documentation](#documentation)
* [Contributing](#contributing)
* [License](#license)



## Getting started

### Prerequisites

- Python 3.6 (or higher)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the latest release of the package using [pypi](https://pypi.org/project/python-doctr/) as follows:

```shell
pip install python-doctr
```

Or you can install it from source:

```shell
git clone https://github.com/mindee/doctr.git
pip install -e doctr/.
```


## Usage

### Python package

You can use the library like any other python package to analyze your documents as follows:

```python
from doctr.documents import read_pdf, read_img
from doctr.models import ocr_db_crnn_vgg

model = ocr_db_crnn_vgg(pretrained=True)
# PDF
doc = read_pdf("path/to/your/doc.pdf")
result = model([doc])
# Image
page = read_img("path/to/your/img.jpg")
result = model([[page]])
# Export
json_output = result[0].export()
```

For an exhaustive list of pretrained models available, please refer to the [documentation](https://mindee.github.io/doctr/models.html).

### Docker container

If you are to deploy containerized environments, you can use the provided Dockerfile to build a docker image:

```shell
docker build . -t <YOUR_IMAGE_TAG>
```

### Example script

An example script is provided for a simple documentation analysis of a PDF file:

```shell
python scripts/analyze.py path/to/your/doc.pdf
```
All script arguments can be checked using `python scripts/analyze.py --help`

### Demo app

A minimal demo app is provided for you to play with the text detection model!

You will need an extra dependency ([Streamlit](https://streamlit.io/)) for the app to run:
```shell
pip install -r demo/requirements.txt
```
You can then easily run your app in your default browser by running:

```shell
streamlit run demo/app.py
```


## Documentation

The full package documentation is available [here](https://mindee.github.io/doctr/) for detailed specifications. The documentation was built with [Sphinx](https://www.sphinx-doc.org/) using a [theme](github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](https://readthedocs.org/).



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
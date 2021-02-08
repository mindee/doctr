
# DocTR: Document Text Recognition

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![CircleCI](https://circleci.com/gh/minde/doctr.svg?style=shield&circle-token=12c96bf5500b9dbe98f4ea0e43ca9c109c7506fe)](https://app.circleci.com/pipelines/github/minde/doctr) [![codecov](https://codecov.io/gh/minde/doctr/branch/main/graph/badge.svg?token=577MO567NM)](https://codecov.io/gh/minde/doctr) [![CodeFactor](https://www.codefactor.io/repository/github/minde/doctr/badge?s=bae07db86bb079ce9d6542315b8c6e70fa708a7e)](https://www.codefactor.io/repository/github/minde/doctr) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/340a76749b634586a498e1c0ab998f08)](https://app.codacy.com/gh/minde/doctr?utm_source=github.com&utm_medium=referral&utm_content=minde/doctr&utm_campaign=Badge_Grade)

Extract valuable information from your documents.



## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [Python package](#python-package)
  * [Docker container](#docker-container)
* [Contributing](#contributing)
* [License](#license)



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

Clone the project and install it:

```shell
git clone https://github.com/minde/doctr.git
pip install -e doctr/.
```



## Usage

### Python package

You can use the library like any other python package to analyze your documents as follows:

```python
from doctr.documents import read_pdf
doc = read_pdf("path/to/your/doc.pdf")
```

### Docker container

If you are to deploy containerized environments, you can use the provided Dockerfile to build a docker image:

```shell
docker build . -t <YOUR_IMAGE_TAG>
```



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
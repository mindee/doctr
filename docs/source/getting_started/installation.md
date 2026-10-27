# Installation

docTR is distributed on PyPI as `python-doctr`. It requires **Python 3.11 or newer** and installs
[PyTorch](https://pytorch.org) as its deep learning backend.

## Install the package

::::{tab-set}

:::{tab-item} pip
```bash
pip install python-doctr
```
:::

:::{tab-item} uv
```bash
uv add python-doctr
# or, outside of a uv project
uv pip install python-doctr
```
:::

:::{tab-item} From source
Install the latest development version, e.g. to try a feature that is not released yet:

```bash
pip install "python-doctr @ git+https://github.com/mindee/doctr.git"
```
:::
::::

To check that everything works, print the installed version:

```bash
python -c "import doctr; print(doctr.__version__)"
```

:::{note}
The pretrained weights are **not** part of the package: each model is downloaded the first time you use it and
cached in `~/.cache/doctr` (see [](../deployment/configuration.md) to change this location or to run offline).
:::

## Optional extras

The core package is kept light. A few features need extra dependencies, which you can pull in with *extras*:

| Extra     | Installs                      | Needed for                                                                                       |
| --------- | ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `viz`     | `matplotlib`, `mplcursors`    | Displaying predictions with {meth}`Document.show() <doctr.io.Document.show>`                       |
| `html`    | `weasyprint`                  | Reading web pages with {meth}`DocumentFile.from_url() <doctr.io.DocumentFile.from_url>`            |
| `contrib` | `onnxruntime`                 | The [artefact detector](../user_guide/artefacts.md) of the `contrib` module                     |

```bash
# One extra
pip install "python-doctr[viz]"
# Several extras
pip install "python-doctr[viz,html,contrib]"
```

:::{admonition} WeasyPrint system libraries
:class: dropdown

The `html` extra relies on [WeasyPrint](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation),
which needs Pango on the system. On Debian/Ubuntu: `sudo apt install libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0`.
On macOS: `brew install pango`.
:::

## GPU support

docTR runs on whatever device PyTorch supports. The default `torch` wheel from PyPI already supports NVIDIA GPUs on
Linux; for another CUDA version, ROCm or a CPU-only build, install PyTorch first by following the
[official selector](https://pytorch.org/get-started/locally/), then install docTR:

```bash
# Example: CPU-only build of PyTorch (smaller download)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install python-doctr
```

Check which accelerator PyTorch sees:

```python
import torch

print("CUDA:", torch.cuda.is_available())
print("Apple MPS:", torch.backends.mps.is_available())
```

Models run on the CPU until you move them: see [](../deployment/performance.md) to run them on a GPU or in half precision.

## Docker images

Ready-to-use images with docTR preinstalled are published on the
[GitHub Container Registry](https://github.com/mindee/doctr/pkgs/container/doctr). They are GPU-ready (CUDA 12.2):

```bash
docker run -it --gpus all ghcr.io/mindee/doctr:<tag> bash
```

Pick a `<tag>` from the registry; [](../deployment/docker.md) explains the tag naming scheme and how to build your own image.

## Development installation

To contribute to docTR, clone the repository and install it in editable mode with all the development tools:

```bash
git clone https://github.com/mindee/doctr.git
cd doctr
pip install -e ".[dev]"
pre-commit install
```

See the {doc}`contribution guide <../contributing/contributing>` for the tests, the code style and how to build this documentation.

## Next steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`play;1.1em` Quickstart
:link: quickstart
:link-type: doc

Extract the text of your first document.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Core concepts
:link: concepts
:link-type: doc

Understand predictors, documents and coordinates.
:::
::::

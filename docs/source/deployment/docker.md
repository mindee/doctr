# Docker

Prebuilt images with docTR installed are published on the
[GitHub Container Registry](https://github.com/mindee/doctr/pkgs/container/doctr). They are based on CUDA 12.2 and
ready for GPU inference.

## Run an image

```bash
docker run -it --gpus all ghcr.io/mindee/doctr:<tag> bash
```

Drop `--gpus all` to run on the CPU. For GPU support, your host needs:

- an NVIDIA driver supporting **CUDA 12.2 or newer**, otherwise PyTorch cannot initialize the GPU;
- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  so that Docker can access the GPU.

## Tags

Tags follow the pattern `<deps>-py<python_version>-<version>`:

`<deps>`
: `torch` for the core package, or `torch-viz-html-contrib` with all the optional extras.

`<python_version>`
: The Python version of the image, e.g. `3.11.15`, `3.12.9` or `3.13.14`.

`<version>`
: A release tag (e.g. `v1.1.0`), or a `YYYY-MM` date for the images rebuilt from the `main` branch every three
  months.

For instance, `torch-viz-html-contrib-py3.11.15-v1.1.0` is release 1.1.0 with all the extras on Python 3.11.15.
Browse the [registry](https://github.com/mindee/doctr/pkgs/container/doctr) for the available tags.

## Build your own image

The [`Dockerfile`](https://github.com/mindee/doctr/blob/main/Dockerfile) at the root of the repository accepts
build arguments to choose the Python version, the extras and the docTR version:

```bash
docker build -t doctr \
  --build-arg FRAMEWORK="torch,viz,html,contrib" \
  --build-arg PYTHON_VERSION=3.11.15 \
  --build-arg DOCTR_VERSION=v1.1.0 \
  .
```

| Argument         | Default    | Description                                                 |
| ---------------- | ---------- | ----------------------------------------------------------- |
| `FRAMEWORK`      | `torch`    | Extras to install, comma-separated (e.g. `torch,viz,html`)  |
| `PYTHON_VERSION` | `3.11.15`  | Python version, compiled from source                        |
| `DOCTR_VERSION`  | `main`     | Git branch, tag or commit of docTR to install              |

:::{tip}
Pretrained weights are downloaded at runtime. To avoid it, warm the cache when building the image, e.g. with
`RUN python -c "from doctr.models import ocr_predictor; ocr_predictor(pretrained=True)"`, or mount a volume on
the cache directory (see [](configuration.md)).
:::

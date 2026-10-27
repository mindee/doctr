# Get started

New to docTR? These three pages take you from `pip install` to structured OCR results, and explain the
handful of concepts that the rest of the documentation builds on.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Installation
:link: installation
:link-type: doc

Install the package, the optional extras, or use the Docker images.
:::

:::{grid-item-card} {octicon}`play;1.1em` Quickstart
:link: quickstart
:link-type: doc

Run OCR on your first document and export the result in five minutes.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Core concepts
:link: concepts
:link-type: doc

How the pipeline works, what a `Document` is made of, and how coordinates are expressed.
:::
::::

:::{tip}
Prefer to explore in a notebook? The [Quicktour notebook](https://colab.research.google.com/github/mindee/notebooks/blob/main/doctr/quicktour.ipynb)
runs in Google Colab without installing anything, and the [live demo](https://huggingface.co/spaces/mindee/doctr)
lets you try the models on your own files in the browser.
:::

```{toctree}
:hidden:

installation
quickstart
concepts
```

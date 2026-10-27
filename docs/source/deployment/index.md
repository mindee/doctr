# Deployment

Taking docTR to production means making it fast, packaging it, and serving it. This section covers each step.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Performance
:link: performance
:link-type: doc

GPU and Apple Silicon, half precision, `torch.compile` and other speed-ups.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` ONNX & OnnxTR
:link: onnx
:link-type: doc

Export models to ONNX and run them without PyTorch.
:::

:::{grid-item-card} {octicon}`container;1.1em` Docker
:link: docker
:link-type: doc

Prebuilt GPU-ready images, tags, and building your own.
:::

:::{grid-item-card} {octicon}`globe;1.1em` Web API & demo app
:link: web_api
:link-type: doc

A FastAPI template to serve docTR, and a Streamlit demo.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Configuration
:link: configuration
:link-type: doc

Environment variables, model cache, offline use and AWS Lambda.
:::
::::

```{toctree}
:hidden:

performance
onnx
docker
web_api
configuration
```

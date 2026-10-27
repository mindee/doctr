# Training

The pretrained models work well on common printed documents. When your documents are different (another
language, handwriting, unusual fonts, specific layouts, fields to extract), **fine-tuning** a model on a few hundred
annotated samples usually brings a large improvement.

## The workflow

```{raw} html
<ol class="steps">
  <li><strong>Collect and annotate</strong> a representative sample of your documents, e.g. with the
  <a href="../community/tools.html#doctr-labeler">docTR Labeler</a>.</li>
  <li><strong>Format the data</strong> as an <code>images/</code> folder and a <code>labels.json</code> file,
  following the <a href="train.html">format of each task</a>.</li>
  <li><strong>Train</strong> with the reference script of the task, starting from the pretrained weights.</li>
  <li><strong>Evaluate</strong> the checkpoint on held-out data with the evaluation script.</li>
  <li><strong>Load</strong> your checkpoint into a predictor, alongside the pretrained models.</li>
  <li><strong>Share</strong> it on the Hugging Face Hub, if you like.</li>
</ol>
```

## In this section

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Datasets
:link: datasets
:link-type: doc

Built-in public datasets, synthetic data generators and loaders for your own data.
:::

:::{grid-item-card} {octicon}`mortar-board;1.1em` Training a model
:link: train
:link-type: doc

Data formats, training scripts, multi-GPU and mixed precision, for every task.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Using your trained models
:link: custom_models
:link-type: doc

Load your checkpoints into the OCR, KIE and task-specific predictors.
:::

:::{grid-item-card} {octicon}`share;1.1em` Sharing models
:link: sharing_models
:link-type: doc

Publish and load models with the Hugging Face Hub, and browse community models.
:::
::::

```{toctree}
:hidden:

datasets
train
custom_models
sharing_models
```

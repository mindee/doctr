---
myst:
  html_meta:
    description: "docTR: state-of-the-art OCR for PDFs and images in a few lines of Python. Text detection, text recognition, layout analysis and table structure recognition, powered by PyTorch."
---

# docTR: Document Text Recognition

```{raw} html
<section class="doctr-hero">
  <div class="doctr-hero__grid">
    <div>
      <div class="doctr-hero__eyebrow">Open-source OCR &middot; Powered by PyTorch</div>
      <p class="doctr-hero__title">OCR made <span>seamless</span> &amp; accessible to anyone</p>
      <p class="doctr-hero__lead">
        docTR reads PDFs and images and gives you back every word with its position, organized into
        lines, blocks, layout regions and tables. Run state-of-the-art models in three lines of Python,
        fine-tune them on your own documents and ship them anywhere.
      </p>
      <div class="doctr-hero__actions">
        <a class="doctr-btn doctr-btn--primary" href="getting_started/quickstart.html">Get started &rarr;</a>
        <a class="doctr-btn doctr-btn--ghost" href="user_guide/index.html">User guide</a>
        <a class="doctr-btn doctr-btn--ghost" href="https://github.com/mindee/doctr">GitHub</a>
      </div>
      <div class="doctr-hero__install">
        <span><span class="prompt">$</span> pip install python-doctr</span>
        <button type="button" data-copy="pip install python-doctr" aria-label="Copy the install command">Copy</button>
      </div>
    </div>
    <div class="doctr-window" aria-label="Example: run OCR on a PDF">
      <div class="doctr-window__bar"><i></i><i></i><i></i><span>ocr.py</span></div>
<pre><span class="tk-k">from</span> doctr.io <span class="tk-k">import</span> DocumentFile
<span class="tk-k">from</span> doctr.models <span class="tk-k">import</span> ocr_predictor

model <span class="tk-o">=</span> <span class="tk-f">ocr_predictor</span>(pretrained<span class="tk-o">=</span><span class="tk-n">True</span>)
doc <span class="tk-o">=</span> DocumentFile.<span class="tk-f">from_pdf</span>(<span class="tk-s">"invoice.pdf"</span>)
result <span class="tk-o">=</span> <span class="tk-f">model</span>(doc)

<span class="tk-c"># or export to JSON, Markdown, HTML...</span>
<span class="tk-f">print</span>(result.<span class="tk-f">render</span>())</pre>
      <div class="doctr-window__output"><b>ACME Office Supplies</b>

<b>INVOICE</b>

42 Harbour Road, Hamburg

No. 2026-0412
Date: 26/10/2026
…</div>
    </div>
  </div>
  <div class="doctr-stats">
    <div><strong>4 tasks</strong><span>detection &middot; recognition &middot; layout &middot; tables</span></div>
    <div><strong>20+</strong><span>pretrained architectures</span></div>
    <div><strong>100+</strong><span>built-in vocabularies</span></div>
    <div><strong>6</strong><span>export formats</span></div>
    <div><strong>Apache 2.0</strong><span>free for commercial use</span></div>
  </div>
</section>
```

```{raw} html
<h2 class="doctr-section-title">Everything you need to read documents</h2>
<p class="doctr-section-lead">From a scanned receipt to a hundred-page report: docTR covers the whole pipeline, and every stage can be swapped, tuned or retrained.</p>
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` Two-stage OCR
:link: user_guide/ocr_predictor
:link-type: doc

A text detection model finds the words, a recognition model reads them. Mix and match 9 detection and 9 recognition architectures.
:::

:::{grid-item-card} {octicon}`table;1.1em` Layout & tables
:link: user_guide/layout_and_tables
:link-type: doc

Detect titles, paragraphs, figures and tables, and turn tables into a row/column grid that loads straight into pandas.
:::

:::{grid-item-card} {octicon}`sync;1.1em` Rotated documents
:link: user_guide/rotated_documents
:link-type: doc

Handle skewed scans, upside-down pages and vertical text with built-in page and word orientation classifiers.
:::

:::{grid-item-card} {octicon}`file-code;1.1em` Export anywhere
:link: user_guide/exporting
:link-type: doc

Get plain text, JSON, Markdown, AsciiDoc, HTML or hOCR, in a reading order that follows columns, headers and footers.
:::

:::{grid-item-card} {octicon}`mortar-board;1.1em` Train & share
:link: training/index
:link-type: doc

Fine-tune any model on your own data with the reference training scripts, then share it on the Hugging Face Hub.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` Production ready
:link: deployment/index
:link-type: doc

GPU & half precision, `torch.compile`, ONNX export, Docker images, a CLI and a REST API template.
:::
::::

```{raw} html
<h2 class="doctr-section-title">How it works</h2>
<p class="doctr-section-lead">An OCR predictor chains specialized models. Each stage is a regular PyTorch module you can replace with your own.</p>
```

```{raw} html
:file: _includes/pipeline.html
```

```{raw} html
<h2 class="doctr-section-title">Find your way around</h2>
<p class="doctr-section-lead">Whether you need text out of a PDF in five minutes or want to train a model for a new alphabet, start here.</p>
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`play;1.1em` Quickstart
:link: getting_started/quickstart
:link-type: doc

Install docTR and extract the text of your first document in five minutes.
:::

:::{grid-item-card} {octicon}`book;1.1em` User guide
:link: user_guide/index
:link-type: doc

Load documents, configure the OCR predictor, navigate and export the results.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Model zoo
:link: models/index
:link-type: doc

All architectures with their benchmarks, to pick the right speed/accuracy trade-off.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Training
:link: training/index
:link-type: doc

Datasets, data formats, training scripts and how to load your own checkpoints.
:::

:::{grid-item-card} {octicon}`server;1.1em` Deployment
:link: deployment/index
:link-type: doc

Speed up inference, export to ONNX, and serve docTR with Docker or a REST API.
:::

:::{grid-item-card} {octicon}`code-square;1.1em` API reference
:link: api/index
:link-type: doc

Every public class and function, generated from the source code.
:::
::::

```{raw} html
<div class="doctr-cta">
  <div>
    <h3>Need help with a complex use case?</h3>
    <p>docTR was created by Mindee and is developed and maintained by <a href="https://www.text2knowledge.de">t2k GmbH</a>,
    who offer custom development, integration and consulting around docTR.</p>
  </div>
  <a class="doctr-btn doctr-btn--primary" href="community/index.html#professional-support">Get support &rarr;</a>
</div>
```

```{toctree}
:hidden:

getting_started/index
user_guide/index
models/index
training/index
deployment/index
api/index
community/index
```

# docTR documentation

The source of [mindee.github.io/doctr](https://mindee.github.io/doctr), built with [Sphinx](https://www.sphinx-doc.org),
the [PyData theme](https://pydata-sphinx-theme.readthedocs.io), [MyST Markdown](https://myst-parser.readthedocs.io)
and [sphinx-design](https://sphinx-design.readthedocs.io).

## Build it locally

```bash
# From the root of the repository
pip install -e ".[viz,html,docs]"
make docs-single-version
```

Then open `docs/_build/index.html` in your browser. Sphinx only rebuilds the pages that changed: delete
`docs/_build` to force a full rebuild. For a live preview while you write, `pip install sphinx-autobuild` and run
`sphinx-autobuild docs/source docs/_build`.

`make docs` builds all the deployed versions (see `build.sh`), as the CI does.

## Structure

| Folder | Section | Content |
| --- | --- | --- |
| `source/index.md` | Landing page | Hero, features and entry points |
| `source/getting_started/` | Get started | Installation, quickstart, core concepts |
| `source/user_guide/` | User guide | One page per feature, task-oriented |
| `source/models/` | Model zoo | Architectures, benchmarks and vocabularies |
| `source/training/` | Training | Datasets, training scripts, custom and shared models |
| `source/deployment/` | Deployment | Performance, ONNX, Docker, web API, configuration |
| `source/api/` | API reference | Generated from the docstrings with `autodoc` |
| `source/community/` | Community | Help, tools, resources, contributing and changelog |
| `source/_static/` | | CSS (`css/doctr.css`), JS, fonts, images and the version switcher (`switcher.json`) |
| `source/_templates/` | | Footer components |
| `source/_includes/` | | Snippets included in several pages (pipeline diagram) |

Each section has an `index.md` with an overview and the `toctree` of its pages. The top navigation bar lists the
sections of the `toctree` of `source/index.md`.

## Writing pages

Pages are written in [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html), a superset
of CommonMark that supports all Sphinx features:

````markdown
# Page title

A first paragraph that says what the page is about: it is displayed as a lead.

## Section

Link to another page: [](../user_guide/exporting.md), or to a section: [](exporting.md#figures).
Link to the API: {func}`~doctr.models.ocr_predictor`, {class}`~doctr.io.Document`.

```python
from doctr.models import ocr_predictor
```

:::{tip}
Admonitions: note, tip, important, warning, seealso, dropdown...
:::

::::{tab-set}
:::{tab-item} pip
...
:::
::::
````

Guidelines:

- **Show, then explain**: start sections with a short runnable snippet, then describe the options.
- **Keep snippets runnable**: complete imports, realistic file names, and real outputs. The outputs shown in the
  guides come from `source/_static/images/examples/invoice.pdf`.
- **One topic per page**, and link to the related pages instead of repeating them.
- **Cards** (`grid-item-card`) are for navigation, **tabs** for alternatives (e.g. CPU / GPU), **dropdowns** for
  details most readers can skip.
- Document new public functions and classes in the docstrings (Google style), and add them to the matching page of
  `source/api/`.

## Releasing a new version

1. Add the release to `source/changelog.md`.
2. Add it to `source/_static/switcher.json` and move the `(stable)` label and `"preferred": true` to it.
3. In `build.sh`, build the previous stable release into its own folder and the new one at the root.

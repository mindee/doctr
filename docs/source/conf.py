# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Sphinx configuration of the docTR documentation.

Build it locally from the repository root with ``make docs-single-version`` (output in ``docs/_build``).
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))
import doctr

# -- Project information -----------------------------------------------------

master_doc = "index"
project = "docTR"
_copyright_str = f"-{datetime.now().year}" if datetime.now().year > 2021 else ""
copyright = f"2021{_copyright_str}, Mindee"
author = "François-Guillaume Fernandez, Charles Gaillard, Olivier Dulcy, Felix Dittrich"

# The full version, including alpha/beta/rc tags
version = doctr.__version__
release = doctr.__version__ + "-git"

# Version selected in the version switcher. `build.sh` sets it for every deployed version
# ("latest" for the main branch, "v1.1.0", ... for the releases).
DOCS_VERSION = os.environ.get("DOCTR_DOCS_VERSION", "latest")

# `build.sh` also renders the previous releases with this configuration: their sources still rely on
# a few extensions that the current pages do not use anymore.
LEGACY_BUILD = os.environ.get("DOCTR_DOCS_LEGACY", "0") == "1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

if LEGACY_BUILD:
    extensions += ["sphinx.ext.autosectionlabel", "sphinxemoji.sphinxemoji", "sphinx_tabs.tabs"]
else:
    extensions += ["sphinx_reredirects"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "notebooks/*.rst", "**/_generated", "_includes"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "pypdfium2": ("https://pypdfium2.readthedocs.io/en/stable/", None),
}

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_substitutions = {
    "version": version,
}

# -- API documentation -------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True
add_module_names = False
python_maximum_signature_line_length = 88
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
napoleon_use_rtype = False

# -- Copy button -------------------------------------------------------------

# Strip the prompts of doctest / shell snippets when copying them
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_exclude = ".linenos, .gp, .go"

# -- Redirects (pages moved by the documentation refactor) -------------------

redirects = {
    "getting_started/installing": "installation.html",
    "notebooks": "community/notebooks.html",
    "using_doctr/using_models": "../user_guide/ocr_predictor.html",
    "using_doctr/using_datasets": "../training/datasets.html",
    "using_doctr/using_contrib_modules": "../user_guide/artefacts.html",
    "using_doctr/sharing_models": "../training/sharing_models.html",
    "using_doctr/using_model_export": "../deployment/performance.html",
    "using_doctr/custom_models_training": "../training/train.html",
    "using_doctr/running_on_aws": "../deployment/configuration.html",
    "using_doctr/using_cli": "../user_guide/cli.html",
    "modules/contrib": "../api/contrib.html",
    "modules/datasets": "../api/datasets.html",
    "modules/io": "../api/io.html",
    "modules/models": "../api/models.html",
    "modules/transforms": "../api/transforms.html",
    "modules/utils": "../api/utils.html",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "docTR documentation"
html_short_title = "docTR"
html_favicon = "_static/images/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["css/doctr.css"]
html_js_files = ["js/doctr.js"]
html_show_sourcelink = False
html_copy_source = False

html_context = {
    "github_user": "mindee",
    "github_repo": "doctr",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}

html_theme_options = {
    "logo": {
        "image_light": "_static/images/logo-light.png",
        "image_dark": "_static/images/logo-dark.png",
        "alt_text": "docTR - Home",
    },
    "navbar_align": "left",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 4 if LEGACY_BUILD else 7,
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": DOCS_VERSION,
    },
    "show_version_warning_banner": not LEGACY_BUILD,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mindee/doctr",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/python-doctr/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Live demo on Hugging Face",
            "url": "https://huggingface.co/spaces/mindee/doctr",
            "icon": "fa-solid fa-wand-magic-sparkles",
        },
    ],
    "use_edit_page_button": not LEGACY_BUILD,
    "secondary_sidebar_items": {
        "**": ["page-toc", "edit-this-page"],
        "index": [],
    },
    "show_toc_level": 2,
    "show_nav_level": 1,
    "navigation_depth": 3,
    "show_prev_next": True,
    "back_to_top_button": True,
    "search_as_you_type": True,
    "footer_start": ["copyright"],
    "footer_center": ["maintainers"],
    "footer_end": ["license"],
    "pygments_light_style": "github-light",
    "pygments_dark_style": "github-dark",
    "analytics": {"google_analytics_id": "G-40DVRMX8T4"},
}

# The landing page is a full-width page
html_sidebars = {"index": []}

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Generated content ---------------------------------------------------------

# Vocabularies listed on `models/vocabs.md`, grouped by script. Any vocabulary missing from these groups is listed
# under "Other scripts", so that the page always reflects `doctr.datasets.VOCABS`.
VOCAB_GROUPS = {
    "General purpose": ["latin", "english", "multilingual", "latex"],
    "Latin script": [
        "albanian",
        "afrikaans",
        "azerbaijani",
        "basque",
        "bosnian",
        "catalan",
        "croatian",
        "czech",
        "danish",
        "dutch",
        "estonian",
        "esperanto",
        "french",
        "finnish",
        "frisian",
        "galician",
        "german",
        "hausa",
        "hungarian",
        "icelandic",
        "indonesian",
        "irish",
        "italian",
        "latvian",
        "lithuanian",
        "luxembourgish",
        "malagasy",
        "malay",
        "maltese",
        "maori",
        "montenegrin",
        "norwegian",
        "polish",
        "portuguese",
        "quechua",
        "romanian",
        "scottish_gaelic",
        "serbian_latin",
        "slovak",
        "slovene",
        "somali",
        "spanish",
        "swahili",
        "swedish",
        "tagalog",
        "turkish",
        "uzbek_latin",
        "vietnamese",
        "welsh",
        "yoruba",
        "zulu",
    ],  # fmt: skip
    "Cyrillic script": [
        "russian",
        "belarusian",
        "tatar",
        "tajik",
        "kazakh",
        "kyrgyz",
        "bulgarian",
        "macedonian",
        "mongolian",
        "yakut",
        "serbian_cyrillic",
        "uzbek_cyrillic",
        "ukrainian",
    ],  # fmt: skip
    "Greek script": ["greek", "greek_extended"],
    "Hebrew & Arabic scripts": ["hebrew", "arabic", "persian", "urdu", "pashto", "kurdish", "uyghur", "sindhi"],
    "Indic scripts": [
        "devanagari",
        "hindi",
        "sanskrit",
        "marathi",
        "nepali",
        "gujarati",
        "bengali",
        "tamil",
        "telugu",
        "kannada",
        "sinhala",
        "malayalam",
        "punjabi",
        "odia",
    ],  # fmt: skip
    "Southeast Asian & other scripts": [
        "thai",
        "lao",
        "khmer",
        "burmese",
        "javanese",
        "sudanese",
        "armenian",
        "georgian",
        "ethiopic",
    ],  # fmt: skip
    "East Asian scripts": ["japanese", "korean", "simplified_chinese"],
}


def _generate_vocab_tables(app):
    """Write the tables of the supported vocabularies, included by `models/vocabs.md`"""
    from html import escape

    from doctr.datasets import VOCABS

    english = set(VOCABS["english"])
    listed = {name for names in VOCAB_GROUPS.values() for name in names}
    first_lang = list(VOCABS).index("latin")  # the entries before are building blocks (digits, punctuation, ...)
    others = [name for name in list(VOCABS)[first_lang:] if name not in listed]
    groups = {**VOCAB_GROUPS, **({"Other scripts": others} if others else {})}

    parts = []
    for title, names in groups.items():
        rows = []
        for name in names:
            if name not in VOCABS:
                continue
            chars = VOCABS[name]
            extra = "".join(c for c in chars if c not in english)
            if title == "Latin script" and len(extra) < len(chars) / 2:
                summary = f'<span class="vocab-base">english +</span> {escape(extra)}' if extra else "english"
            else:
                summary = escape(chars[:60]) + ("…" if len(chars) > 60 else "")
            rows.append(
                f"<tr><td><code>{name}</code></td><td>{len(chars)}</td>"
                f'<td><details><summary>{summary}</summary><div class="vocab-chars">{escape(chars)}</div>'
                "</details></td></tr>"
            )
        parts.append(
            f"### {title}\n\n```{{raw}} html\n"
            '<div class="pst-scrollable-table-container"><table class="table vocab-table">'
            "<thead><tr><th>Name</th><th>Size</th><th>Characters</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>\n```\n"
        )

    out_dir = os.path.join(app.srcdir, "models", "_generated")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocabs.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def setup(app):
    if not LEGACY_BUILD:
        app.connect("builder-inited", _generate_vocab_tables)

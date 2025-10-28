# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
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

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinxemoji.sphinxemoji",  # cf. https://sphinxemojicodes.readthedocs.io/en/stable/
    "sphinx_copybutton",
    "recommonmark",
    "sphinx_markdown_tables",
    "sphinx_tabs.tabs",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pypdfium2": ("https://pypdfium2.readthedocs.io/en/stable/", None),
}

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "notebooks/*.rst"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"
pygments_dark_style = "monokai"
highlight_language = "python3"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-sidebar-background": "#082747",
        "color-sidebar-background-border": "#082747",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
        "color-sidebar-link-text": "white",
        "sidebar-caption-font-size": "normal",
        "color-sidebar-item-background--hover": " #5dade2",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#1a1c1e",
        "color-sidebar-background-border": "#1a1c1e",
        "color-sidebar-caption-text": "white",
        "color-sidebar-link-text--top-level": "white",
    },
}

html_logo = "_static/images/Logo-docTR-white.png"
html_favicon = "_static/images/favicon.ico"
html_title = "docTR documentation"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


def add_ga_javascript(app, pagename, templatename, context, doctree):
    # Add googleanalytics id
    # ref: https://github.com/orenhecht/googleanalytics/blob/master/sphinxcontrib/googleanalytics.py

    metatags = context.get("metatags", "")
    metatags += """
    <!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id={0}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{0}');
</script>
    """.format(app.config.googleanalytics_id)
    context["metatags"] = metatags


def setup(app):
    app.add_config_value("googleanalytics_id", "G-40DVRMX8T4", "html")
    app.add_css_file("css/mindee.css")
    app.add_js_file("js/custom.js")
    app.connect("html-page-context", add_ga_javascript)

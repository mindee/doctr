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

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../..'))
import doctr

# -- Project information -----------------------------------------------------

master_doc = 'index'
project = 'docTR'
_copyright_str = f"-{datetime.now().year}" if datetime.now().year > 2021 else ""
copyright = f"2021{_copyright_str}, Mindee"
author = 'François-Guillaume Fernandez, Charles Gaillard'

# The full version, including alpha/beta/rc tags
version = doctr.__version__
release = doctr.__version__ + '-git'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'sphinxemoji.sphinxemoji',  # cf. https://sphinxemojicodes.readthedocs.io/en/stable/
    'sphinx_copybutton',
    'recommonmark',
    'sphinx_markdown_tables',
]

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [u'_build', 'Thumbs.db', '.DS_Store', 'notebooks/*.rst']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'
highlight_language = 'python3'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'collapse_navigation': False,
    'display_version': False,
    'logo_only': False,
    'analytics_id': 'G-40DVRMX8T4',
}

html_logo = '_static/images/Logo-docTR-white.png'
html_favicon = '_static/images/favicon.ico'



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

def setup(app):
    app.add_css_file('css/mindee.css')
    app.add_js_file('js/custom.js')

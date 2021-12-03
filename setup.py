# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Package installation setup
"""

import os
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

version = "0.5.0a0"
sha = 'Unknown'
src_folder = 'doctr'
package_index = 'python-doctr'

cwd = Path(__file__).parent.absolute()

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass
    version += '+' + sha[:7]
print(f"Building wheel {package_index}-{version}")

with open(cwd.joinpath(src_folder, 'version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")

with open('README.md', 'r') as f:
    readme = f.read()

# Borrowed from https://github.com/huggingface/transformers/blob/master/setup.py
_deps = [
    "importlib_metadata",
    "numpy>=1.16.0",
    "scipy>=1.4.0",
    "opencv-python>=3.4.5.20",
    "tensorflow>=2.4.0",
    "PyMuPDF>=1.16.0,!=1.18.11,!=1.18.12",  # 18.11 and 18.12 fail (issue #222)
    "pyclipper>=1.2.0",
    "shapely>=1.6.0",
    "matplotlib>=3.1.0,<3.4.3",
    "mplcursors>=0.3",
    "weasyprint>=52.2,<53.0",
    "unidecode>=1.0.0",
    "tensorflow-cpu>=2.4.0",
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "Pillow>=8.3.2",  # cf. https://github.com/advisories/GHSA-98vv-pw6r-q6q4
    "tqdm>=4.30.0",
    "tensorflow-addons>=0.13.0",
    "rapidfuzz>=1.6.0",
    "keras<2.7.0",
    # Testing
    "pytest>=5.3.2",
    "coverage>=4.5.4",
    "requests>=2.20.0",
    "requirements-parser==0.2.0",
    # Quality
    "flake8>=3.9.0",
    "isort>=5.7.0",
    "mypy>=0.812",
    # Docs
    "sphinx<3.5.0",
    "sphinx-rtd-theme==0.4.3",
    "sphinxemoji>=0.1.8",
    "sphinx-copybutton>=0.3.1",
    "docutils<0.18",
    "recommonmark>=0.7.1",
    "sphinx-markdown-tables>=0.0.15",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


install_requires = [
    deps["importlib_metadata"] + ";python_version<'3.8'",  # importlib_metadata for Python versions that don't have it
    deps["numpy"],
    deps["scipy"],
    deps["opencv-python"],
    deps["PyMuPDF"],
    deps["pyclipper"],
    deps["shapely"],
    deps["matplotlib"],
    deps["mplcursors"],
    deps["weasyprint"],
    deps["unidecode"],
    deps["Pillow"],
    deps["tqdm"],
    deps["rapidfuzz"],
]

extras = {}
extras["tf"] = deps_list(
    "tensorflow",
    "tensorflow-addons",
    "keras",
)

extras["tf-cpu"] = deps_list(
    "tensorflow-cpu",
    "tensorflow-addons",
    "keras",
)

extras["torch"] = deps_list(
    "torch",
    "torchvision",
)

extras["all"] = (
    extras["tf"]
    + extras["torch"]
)

extras["testing"] = deps_list(
    "pytest",
    "coverage",
    "requests",
    "requirements-parser",
)

extras["quality"] = deps_list(
    "flake8",
    "isort",
    "mypy"
)

extras["docs_specific"] = deps_list(
    "sphinx",
    "sphinx-rtd-theme",
    "sphinxemoji",
    "sphinx-copybutton",
    "docutils",
    "recommonmark",
    "sphinx-markdown-tables",
)

extras["docs"] = extras["all"] + extras["docs_specific"]

extras["dev"] = (
    extras["all"]
    + extras["testing"]
    + extras["quality"]
    + extras["docs_specific"]
)

setup(
    # Metadata
    name=package_index,
    version=version,
    author='Mindee',
    author_email='contact@mindee.com',
    maintainer='François-Guillaume Fernandez, Charles Gaillard',
    description='Document Text Recognition (docTR): deep Learning for high-performance OCR on documents.',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/mindee/doctr',
    download_url='https://github.com/mindee/doctr/tags',
    license='Apache',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        "Intended Audience :: Education",
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['OCR', 'deep learning', 'computer vision', 'tensorflow', 'pytorch', 'text detection', 'text recognition'],

    # Package info
    packages=find_packages(exclude=('tests',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras,
    package_data={'': ['LICENSE']}
)

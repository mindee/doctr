# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Package installation setup
"""

import os
import re
from pathlib import Path
import subprocess

from setuptools import find_packages, setup


version = "0.4.0a0"
sha = 'Unknown'
package_name = 'doctr'

cwd = Path(__file__).parent.absolute()

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass
    version += '+' + sha[:7]
print(f"Building wheel {package_name}-{version}")

with open(cwd.joinpath(package_name, 'version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")

with open('README.md', 'r') as f:
    readme = f.read()

# Borrowed from https://github.com/huggingface/transformers/blob/master/setup.py
_deps = [
    "importlib_metadata",
    "numpy>=1.16.0",
    "scipy>=1.4.0",
    "opencv-python>=4.2",
    "tensorflow>=2.4.0",
    "PyMuPDF>=1.16.0,<1.18.11",
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
extras["tf"] = deps_list("tensorflow", "tensorflow-addons")
extras["tf-cpu"] = deps_list("tensorflow-cpu", "tensorflow-addons")
extras["torch"] = deps_list("torch", "torchvision")
extras["all"] = (
    extras["tf"]
    + extras["torch"]
)

setup(
    # Metadata
    name=os.getenv('PKG_INDEX') if os.getenv('PKG_INDEX') else package_name,
    version=version,
    author='Mindee',
    author_email='contact@mindee.com',
    maintainer='François-Guillaume Fernandez, Charles Gaillard',
    description='Document Text Recognition (DocTR): deep Learning for high-performance OCR on documents.',
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
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras,
    package_data={'': ['LICENSE']}
)

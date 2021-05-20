# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Package installation setup
"""

import os
from pathlib import Path
import subprocess

from setuptools import find_packages, setup


version = "0.2.1a0"
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

requirements = [
    "numpy>=1.16.0",
    "scipy>=1.4.0",
    "opencv-python>=4.2",
    "tensorflow>=2.4.0",
    "PyMuPDF>=1.16.0,<1.18.11",
    "pyclipper>=1.2.0",
    "shapely>=1.6.0",
    "matplotlib>=3.1.0",
    "mplcursors>=0.3",
    "weasyprint>=52.2",
    "unidecode>=1.0.0",
]

setup(
    # Metadata
    name=os.getenv('PKG_INDEX') if os.getenv('PKG_INDEX') else package_name,
    version=version,
    author='François-Guillaume Fernandez, Charles Gaillard',
    author_email='fg@mindee.co',
    description='Extract valuable text information from your documents',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/mindee/doctr',
    download_url='https://github.com/mindee/doctr/tags',
    license='Apache',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=['ocr', 'deep learning', 'tensorflow', 'text detection', 'text recognition'],

    # Package info
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=requirements,
    package_data={'': ['LICENSE']}
)

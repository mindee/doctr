# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

import doctr

PROJECT_NAME: str = 'docTR API template'
PROJECT_DESCRIPTION: str = 'Template API for Optical Character Recognition'
VERSION: str = doctr.__version__
DEBUG: bool = os.environ.get('DEBUG', '') != 'False'

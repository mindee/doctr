# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

import doctr

PROJECT_NAME: str = "docTR API template"
PROJECT_DESCRIPTION: str = "Template API for Optical Character Recognition"
VERSION: str = doctr.__version__
DEBUG: bool = os.environ.get("DEBUG", "") != "False"

# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import secrets


PROJECT_NAME: str = 'DocTR API'
PROJECT_DESCRIPTION: str = 'Template API for Optical Character Recognition'
VERSION: str = "0.1.2a0"
DEBUG: bool = os.environ.get('DEBUG', '') != 'False'

SECRET_KEY: str = secrets.token_urlsafe(32)
if DEBUG:
    # To keep the same Auth at every app loading in debug mode and not having to redo the auth.
    debug_secret_key = "000000000000000000000000000000000000"
    SECRET_KEY = debug_secret_key

# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import string
from typing import Dict

__all__ = ['VOCABS']


VOCABS: Dict[str, str] = {
    'digits': string.digits,
    'ascii_letters': string.ascii_letters,
    'punctuation': string.punctuation,
    'currency': '£€¥¢฿',
    'latin': string.digits + string.ascii_letters + string.punctuation + '°',
    'french': string.digits + string.ascii_letters + string.punctuation + '°' + 'àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ' + '£€¥¢฿',
}

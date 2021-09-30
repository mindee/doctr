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
    'ancient_greek': 'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',
}

VOCABS['latin'] = VOCABS['digits'] + VOCABS['ascii_letters'] + VOCABS['punctuation']
VOCABS['english'] = VOCABS['latin'] + '°' + VOCABS['currency']
VOCABS['legacy_french'] = VOCABS['latin'] + '°' + 'àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ' + VOCABS['currency']
VOCABS['french'] = VOCABS['english'] + 'àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ'
VOCABS['portuguese'] = VOCABS['english'] + 'áàâãéêëíïóôõúüçÁÀÂÃÉËÍÏÓÔÕÚÜÇ' + '¡¿'
VOCABS['spanish'] = VOCABS['english'] + 'áéíóúüñÁÉÍÓÚÜÑ' + '¡¿'
VOCABS['german'] = VOCABS['english'] + 'äöüßÄÖÜẞ'

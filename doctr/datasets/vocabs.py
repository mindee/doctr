# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import re
import string

__all__ = ["VOCABS"]


VOCABS: dict[str, str] = {
    # Arabic & Persian
    "arabic_diacritics": "ًٌٍَُِّْ",
    "arabic_digits": "٠١٢٣٤٥٦٧٨٩",
    "arabic_letters": "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي",
    "arabic_punctuation": "؟؛«»—",
    "persian_letters": "پچڢڤگ",
    # Bangla
    "bangla_digits": "০১২৩৪৫৬৭৮৯",
    "bangla_letters": "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃেৈোৌ্ৎংঃঁ",
    # Cyrillic
    "generic_cyrillic_letters": "абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
    "russian_cyrillic_letters": "ёыэЁЫЭ",
    "russian_signs": "ъЪ",
    # Greek
    "ancient_greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    # Gujarati
    "gujarati_consonants": "ખગઘચછજઝઞટઠડઢણતથદધનપફબભમયરલવશસહળક્ષ",
    "gujarati_digits": "૦૧૨૩૪૫૬૭૮૯",
    "gujarati_punctuation": "૰ઽ◌ંઃ॥ૐ઼ઁ" + "૱",
    "gujarati_vowels": "અઆઇઈઉઊઋએઐઓ",
    # Hindi
    "hindi_digits": "०१२३४५६७८९",
    "hindi_letters": "अआइईउऊऋॠऌॡएऐओऔंःकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
    "hindi_punctuation": "।,?!:्ॐ॰॥",
    # Hebrew
    "hebrew_cantillations": "֑֖֛֢֣֤֥֦֧֪֚֭֮֒֓֔֕֗֘֙֜֝֞֟֠֡֨֩֫֬֯",
    "hebrew_letters": "אבגדהוזחטיךכלםמןנסעףפץצקרשת",
    "hebrew_specials": "ׯװױײיִﬞײַﬠﬡﬢﬣﬤﬥﬦﬧﬨ﬩שׁשׂשּׁשּׂאַאָאּבּגּדּהּוּזּטּיּךּכּלּמּנּסּףּפּצּקּרּשּתּוֹבֿכֿפֿﭏ",
    "hebrew_punctuation": "ֽ־ֿ׀ׁׂ׃ׅׄ׆׳״",
    "hebrew_vowels": "ְֱֲֳִֵֶַָׇֹֺֻ",
    # Latin
    "digits": string.digits,
    "ascii_letters": string.ascii_letters,
    "punctuation": string.punctuation,
    "currency": "£€¥¢฿",
}

# Latin & latin-dependent alphabets
VOCABS["latin"] = VOCABS["digits"] + VOCABS["ascii_letters"] + VOCABS["punctuation"]
VOCABS["english"] = VOCABS["latin"] + "°" + VOCABS["currency"]

VOCABS["albanian"] = VOCABS["english"] + "çëÇË"

VOCABS["afrikaans"] = VOCABS["english"] + "èëïîôûêÈËÏÎÔÛÊ"

VOCABS["azerbaijani"] = re.sub(r"[Ww]", "", VOCABS["english"]) + "çəğöşüÇƏĞÖŞÜ" + "₼"

VOCABS["basque"] = VOCABS["english"] + "ñçÑÇ"

VOCABS["bosanski"] = re.sub(r"[QqWwXxYy]", "", VOCABS["english"]) + "čćđšžČĆĐŠŽ"

VOCABS["catalan"] = VOCABS["english"] + "àèéíïòóúüçÀÈÉÍÏÒÓÚÜÇ"

VOCABS["croatian"] = VOCABS["english"] + "ČčĆćĐđŠšŽž"

VOCABS["czech"] = VOCABS["english"] + "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"

VOCABS["danish"] = VOCABS["english"] + "æøåÆØÅ"

VOCABS["dutch"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ"

VOCABS["estonian"] = VOCABS["english"] + "šžõäöüŠŽÕÄÖÜ"

VOCABS["esperanto"] = re.sub(r"[QqWwXxYy]", "", VOCABS["english"]) + "ĉĝĥĵŝŭĈĜĤĴŜŬ" + "₷"

VOCABS["french"] = VOCABS["english"] + "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"
VOCABS["legacy_french"] = VOCABS["latin"] + "°" + "àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ" + VOCABS["currency"]

VOCABS["finnish"] = VOCABS["english"] + "äöÄÖ"

VOCABS["frisian"] = re.sub(r"[QqXx]", "", VOCABS["english"]) + "âêôûúÂÊÔÛÚ" + "ƒ"

VOCABS["galician"] = re.sub(r"[JjKkWw]", "", VOCABS["english"]) + "ñÑçÇ"

VOCABS["german"] = VOCABS["english"] + "äöüßÄÖÜẞ"

VOCABS["hausa"] = re.sub(r"[PpQqVvXx]", "", VOCABS["english"]) + "ɓɗƙƴƁƊƘƳ" + "₦"

VOCABS["hungarian"] = VOCABS["english"] + "áéíóöúüÁÉÍÓÖÚÜ"

VOCABS["icelandic"] = re.sub(r"[CcQqWw]", "", VOCABS["english"]) + "ðáéíóúýþæöÐÁÉÍÓÚÝÞÆÖ"

VOCABS["indonesian"] = VOCABS["english"]

VOCABS["irish"] = VOCABS["english"] + "áéíóúÁÉÍÓÚ"

VOCABS["italian"] = VOCABS["english"] + "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"

VOCABS["latvian"] = re.sub(r"[QqWwXx]", "", VOCABS["english"]) + "āčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ"

VOCABS["lithuanian"] = re.sub(r"[QqWwXx]", "", VOCABS["english"]) + "ąčęėįšųūžĄČĘĖĮŠŲŪŽ"

VOCABS["luxembourgish"] = VOCABS["english"] + "äöüéëÄÖÜÉË"

VOCABS["malagasy"] = re.sub(r"[CcQqUuWwXx]", "", VOCABS["english"]) + "ôñÔÑ"

VOCABS["malay"] = VOCABS["english"]

VOCABS["maltese"] = re.sub(r"[CcYy]", "", VOCABS["english"]) + "ċġħżĊĠĦŻ"

VOCABS["montenegrin"] = re.sub(r"[QqWwXxYy]", "", VOCABS["english"]) + "čćšžźČĆŠŚŽŹ"

VOCABS["norwegian"] = VOCABS["english"] + "æøåÆØÅ"

VOCABS["polish"] = VOCABS["english"] + "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"

VOCABS["portuguese"] = VOCABS["english"] + "áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ"

VOCABS["romanian"] = VOCABS["english"] + "ăâîșțĂÂÎȘȚ"

VOCABS["scottish_gaelic"] = re.sub(r"[JjKkQqVvWwXxYyZz]", "", VOCABS["english"]) + "àèìòùÀÈÌÒÙ"

VOCABS["serbian_latin"] = VOCABS["english"] + "čćđžšČĆĐŽŠ"

VOCABS["slovak"] = VOCABS["english"] + "ôäčďľňšťžáéíĺóŕúýÔÄČĎĽŇŠŤŽÁÉÍĹÓŔÚÝ"

VOCABS["slovene"] = re.sub(r"[QqWwXxYy]", "", VOCABS["english"]) + "čćđšžČĆĐŠŽ"

VOCABS["somali"] = re.sub(r"[PpVvZz]", "", VOCABS["english"])

VOCABS["spanish"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ" + "¡¿"

VOCABS["swahili"] = re.sub(r"[QqXx]", "", VOCABS["english"])

VOCABS["swedish"] = VOCABS["english"] + "åäöÅÄÖ"

VOCABS["tagalog"] = re.sub(r"[CcQqWwXx]", "", VOCABS["english"]) + "ñÑ" + "₱"

VOCABS["turkish"] = re.sub(r"[QqWwXx]", "", VOCABS["english"]) + "çğıöşüâîûÇĞİÖŞÜÂÎÛ"

VOCABS["uzbek_latin"] = re.sub(r"[Ww]", "", VOCABS["english"]) + "çğɉñöşÇĞɈÑÖŞ"

VOCABS["vietnamese"] = (
    VOCABS["english"]
    + "áàảạãăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựíìỉĩịýỳỷỹỵ"
    + "ÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴ"
)

VOCABS["welsh"] = re.sub(r"[KkQqVvXxZz]", "", VOCABS["english"]) + "âêîôŵŷÂÊÎÔŴŶ"

VOCABS["Zulu"] = VOCABS["english"]

# Non-latin alphabets.
# Arabic
VOCABS["arabic"] = (
    VOCABS["digits"]
    + VOCABS["arabic_digits"]
    + VOCABS["arabic_letters"]
    + VOCABS["persian_letters"]
    + VOCABS["arabic_diacritics"]
    + VOCABS["arabic_punctuation"]
    + VOCABS["punctuation"]
)

# Bangla
VOCABS["bangla"] = VOCABS["bangla_letters"] + VOCABS["bangla_digits"]

# Gujarati
VOCABS["gujarati"] = (
    VOCABS["gujarati_vowels"]
    + VOCABS["gujarati_consonants"]
    + VOCABS["gujarati_digits"]
    + VOCABS["gujarati_punctuation"]
    + VOCABS["punctuation"]
)

# Hebrew
VOCABS["hebrew"] = (
    VOCABS["english"]
    + VOCABS["hebrew_letters"]
    + VOCABS["hebrew_vowels"]
    + VOCABS["hebrew_punctuation"]
    + VOCABS["hebrew_cantillations"]
    + VOCABS["hebrew_specials"]
    + "₪"
)

# Hindi
VOCABS["hindi"] = VOCABS["hindi_letters"] + VOCABS["hindi_digits"] + VOCABS["hindi_punctuation"]

# Cyrillic
VOCABS["russian"] = (
    VOCABS["generic_cyrillic_letters"]
    + VOCABS["russian_cyrillic_letters"]
    + VOCABS["russian_signs"]
    + VOCABS["digits"]
    + VOCABS["punctuation"]
    + "₽"
)

VOCABS["ukrainian"] = (
    VOCABS["generic_cyrillic_letters"] + VOCABS["digits"] + VOCABS["punctuation"] + VOCABS["currency"] + "ґіїєҐІЇЄ₴"
)

# Multi-lingual
VOCABS["multilingual"] = "".join(
    dict.fromkeys(
        VOCABS["french"]
        + VOCABS["portuguese"]
        + VOCABS["spanish"]
        + VOCABS["german"]
        + VOCABS["czech"]
        + VOCABS["croatian"]
        + VOCABS["polish"]
        + VOCABS["dutch"]
        + VOCABS["italian"]
        + VOCABS["norwegian"]
        + VOCABS["danish"]
        + VOCABS["finnish"]
        + VOCABS["swedish"]
        + "§"
    )
)

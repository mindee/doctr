# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import string
from typing import Dict

__all__ = ["VOCABS"]


VOCABS: Dict[str, str] = {
    "digits": string.digits,
    "ascii_letters": string.ascii_letters,
    "punctuation": string.punctuation,
    "currency": "£€¥¢฿",
    "ancient_greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    "arabic_letters": "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي",
    "persian_letters": "پچڢڤگ",
    "arabic_digits": "٠١٢٣٤٥٦٧٨٩",
    "arabic_diacritics": "ًٌٍَُِّْ",
    "arabic_punctuation": "؟؛«»—",
    "hindi_letters": "अआइईउऊऋॠऌॡएऐओऔअंअःकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
    "hindi_digits": "०१२३४५६७८९",
    "hindi_punctuation": "।,?!:्ॐ॰॥॰",
    "gujarati_consonants":"કખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલવશષસહષજ્ઞ",
    "gujarati_vowels": "અઆઇઈઉઊઋએઐઓઔઅંઅઃ ",
    "gujarati_digits":"૦૧૨૩૪૫૬૭૮૯",
    "gujarati_diacritics":"""કકાકિકીકુકૂકૃકેકૈકોકૌકંકઃ ખખાખિખીખુખૂખૃખેખૈખોખૌખંખઃ ગગાગિગીગુગૂગૃગેગૈગોગૌગંગઃ ઘઘાઘિઘીઘુઘૂઘૃઘેઘૈઘોઘૌઘંઘઃ ઙઙાઙિઙીઙુઙૂઙૃઙેઙૈઙોઙૌઙંઙઃ ચચાચિચીચુચૂચૃચેચૈચોચૌચંચઃ 
    છછાછિછીછુછૂછૃછેછૈછોછૌછંછઃ જજાજિજીજુજુજૃજેજૈજોજૌજંજઃ ઝઝાઝિઝીઝુઝૂઝૃઝેઝૈઝોઝૌઝંઝઃ ઞઞાઞિઞીઞુઞૂઞૃઞેઞૈઞોઞૌઞંઞઃ ટટાટિટીટુટૂટૃટેટૈટોટૌટંટઃ ઠઠાઠિઠીઠુઠૂઠૃઠેઠૈઠોઠૌઠંઠઃ ડડાડિડીડુડૂડૃડેડૈડોડૌડંડઃ ઢઢાઢિઢીઢુઢૂઢૃઢેઢૈઢોઢૌઢંઢઃ ણણાણિણીણુણૂણૃણેણૈણોણૌણંણઃ 
    તતાતિતીતુતૂતૃતેતૈતોતૌતંતઃ થથાથિથીથુથૂથૃથીથૈથોથૌથંથઃ દદાદિદીદુદૂદૃદેદૈદોદૌદંદઃ ધધાધિધીધુધૂધૃધેધૈધોધૌધંધઃ નનાનિનીનુનૂનૃનેનૈનોનૌનંનઃ પપાપિપીપુપૂપૃપેપૈપોપૌપંપઃ ફફાફિફીફુફૂફૃફેફૈફોફૌફંફઃ બબાબિબીબુબૂબૃબેબૈબોબૌબંબઃ ભભાભિભીભુભૂભૃભેભૈભોભૌભંભઃ 
    મમામિમીમુમૂમૃમેમામોમાયમંમઃ યયાયિયીયુયુયૃયેયૈયોયૌયંયઃ રરારિરીરૂરૃરેરૈરોરૌરંરઃ લલાલિલીલુલૂલૃલેલૈલોલૌલંલઃ વવાવિવીવિવૂવૃવેવૈવોવૈવંવઃ શશાશિશીશુશૂશૃશેશૈશોશૌશંશઃ ષષાષિષીષુષૂષૃષેષૈષોષૌષંષઃ જ્ઞજ્ઞાજ્ઞિજ્ઞીજ્ઞુજ્ઞૂજ્ઞૃજ્ઞેજ્ઞૈજ્ઞોજ્ઞૌજ્ઞંજ્ઞઃ """,
    "gujarati_punctuation":",.!?:;'()[]-_/|\✶૰૱`'",
    "bangla_letters": "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃেৈোৌ্ৎংঃঁ",
    "bangla_digits": "০১২৩৪৫৬৭৮৯",
    "generic_cyrillic_letters": "абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
}

VOCABS["latin"] = VOCABS["digits"] + VOCABS["ascii_letters"] + VOCABS["punctuation"]
VOCABS["english"] = VOCABS["latin"] + "©" +VOCABS["currency"].replace("¥",' ').replace('€','™').replace('¢','®')
VOCABS["legacy_french"] = VOCABS["latin"] + "°" + "àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ" + VOCABS["currency"]
VOCABS["french"] = VOCABS["english"] + "àâéèêëáîôùûüÀÂÉÈÊËÎÏÔÙÛÜÚÇ" 
VOCABS["portuguese"] = VOCABS["english"] + "áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ"
VOCABS["spanish"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ" + "¡¿"
VOCABS["italian"] = VOCABS["english"] + "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"
VOCABS["german"] = VOCABS["english"] + "äöüßÄÖÜẞ"
VOCABS["arabic"] = (
    VOCABS["digits"]
    + VOCABS["arabic_digits"]
    + VOCABS["arabic_letters"]
    + VOCABS["persian_letters"]
    + VOCABS["arabic_diacritics"]
    + VOCABS["arabic_punctuation"]
    + VOCABS["punctuation"]
)
VOCABS["czech"] = VOCABS["english"] + "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"
VOCABS["polish"] = VOCABS["english"] + "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
VOCABS["dutch"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ"
VOCABS["norwegian"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["danish"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["finnish"] = VOCABS["english"] + "äöÄÖ"
VOCABS["swedish"] = VOCABS["english"] + "åäöÅÄÖ"
VOCABS["vietnamese"] = (
    VOCABS["english"]
    + "áàảạãăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựiíìỉĩịýỳỷỹỵ"
    + "ÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰIÍÌỈĨỊÝỲỶỸỴ"
)
VOCABS["hebrew"] = VOCABS["english"] + "אבגדהוזחטיכלמנסעפצקרשת" + "₪"
VOCABS["hindi"] = VOCABS["hindi_letters"] + VOCABS["hindi_digits"] + VOCABS["hindi_punctuation"]
VOCABS["gujarati"] = (         
    VOCABS['gujarati_consonants'] 
    + VOCABS["gujarati_vowels"] 
    + VOCABS['gujarati_digits'] 
    + VOCABS['gujarati_diacritics'] 
    + VOCABS['gujarati_punctuation']
)
VOCABS["bangla"] = VOCABS["bangla_letters"] + VOCABS["bangla_digits"]
VOCABS["ukrainian"] = (
    VOCABS["generic_cyrillic_letters"] + VOCABS["digits"] + VOCABS["punctuation"] + VOCABS["currency"] + "ґіїєҐІЇЄ₴"
)
VOCABS["multilingual"] = "".join(
    dict.fromkeys(
        VOCABS["french"]
        + VOCABS["portuguese"]
        + VOCABS["spanish"]
        + VOCABS["german"]
        + VOCABS["czech"]
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

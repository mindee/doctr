doctr.datasets
==============

.. currentmodule:: doctr.datasets

.. _datasets:

doctr.datasets
--------------

.. autoclass:: FUNSD

.. autoclass:: SROIE

.. autoclass:: CORD

.. autoclass:: IIIT5K

.. autoclass:: SVT

.. autoclass:: SVHN

.. autoclass:: SynthText

.. autoclass:: IC03

.. autoclass:: IC13

.. autoclass:: IMGUR5K

.. autoclass:: MJSynth

.. autoclass:: IIITHWS

.. autoclass:: DocArtefacts

.. autoclass:: WILDRECEIPT

.. autoclass:: COCOTEXT

Synthetic dataset generator
---------------------------

.. autoclass:: CharacterGenerator

.. autoclass:: WordGenerator

Custom dataset loader
---------------------

.. autoclass:: DetectionDataset

.. autoclass:: RecognitionDataset

.. autoclass:: OCRDataset

Dataset utils
-------------

.. autofunction:: translate

.. autofunction:: encode_string

.. autofunction:: decode_sequence

.. autofunction:: encode_sequences

.. autofunction:: pre_transform_multiclass

.. autofunction:: crop_bboxes_from_image

.. autofunction:: convert_target_to_relative

.. _vocabs:

Supported Vocabs
----------------

Since textual content has to be encoded properly for models to interpret them efficiently, docTR supports multiple sets
of vocabs.

.. list-table:: docTR Vocabs
   :widths: 20 5 50
   :header-rows: 1

   * - Name
     - size
     - characters
   * - ascii_letters
     - 52
     - abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
   * - currency
     - 5
     - £€¥¢฿
   * - digits
     - 10
     - 0123456789
   * - punctuation
     - 32
     - !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
   * - albanian
     - 104
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿çëÇË
   * - afrikaans
     - 114
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿èëïîôûêÈËÏÎÔÛÊ
   * - azerbaijani
     - 111
     - 0123456789abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿çəğöşüÇƏĞÖŞÜ₼
   * - basque
     - 104
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ñçÑÇ
   * - bosanski
     - 102
     - 0123456789abcdefghijklmnoprstuvzABCDEFGHIJKLMNOPRSTUVZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿čćđšžČĆĐŠŽ
   * - catalan
     - 120
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿àèéíïòóúüçÀÈÉÍÏÒÓÚÜÇ
   * - croatian
     - 110
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ČčĆćĐđŠšŽž
   * - czech
     - 130
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ
   * - danish
     - 106
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿æøåÆØÅ
   * - dutch
     - 114
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áéíóúüñÁÉÍÓÚÜÑ
   * - english
     - 100
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - estonian
     - 112
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿šžõäöüŠŽÕÄÖÜ
   * - esperanto
     - 105
     - 0123456789abcdefghijklmnoprstuvzABCDEFGHIJKLMNOPRSTUVZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ĉĝĥĵŝŭĈĜĤĴŜŬ₷
   * - french
     - 126
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ
   * - legacy_french
     - 123
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ£€¥¢฿
   * - finnish
     - 104
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿äöÄÖ
   * - frisian
     - 107
     - 0123456789abcdefghijklmnoprstuvwyzABCDEFGHIJKLMNOPRSTUVWYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿âêôûúÂÊÔÛÚƒ
   * - galician
     - 98
     - 0123456789abcdefghilmnopqrstuvxyzABCDEFGHILMNOPQRSTUVXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ñÑçÇ
   * - german
     - 108
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿äöüßÄÖÜẞ
   * - hausa
     - 101
     - 0123456789abcdefghijklmnorstuwyzABCDEFGHIJKLMNORSTUWYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ɓɗƙƴƁƊƘƳ₦
   * - hungarian
     - 114
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áéíóöúüÁÉÍÓÖÚÜ
   * - icelandic
     - 114
     - 0123456789abdefghijklmnoprstuvxyzABDEFGHIJKLMNOPRSTUVXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ðáéíóúýþæöÐÁÉÍÓÚÝÞÆÖ
   * - indonesian
     - 100
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - irish
     - 110
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áéíóúÁÉÍÓÚ
   * - italian
     - 120
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ
   * - latvian
     - 116
     - 0123456789abcdefghijklmnoprstuvyzABCDEFGHIJKLMNOPRSTUVYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿āčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ
   * - lithuanian
     - 112
     - 0123456789abcdefghijklmnoprstuvyzABCDEFGHIJKLMNOPRSTUVYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ąčęėįšųūžĄČĘĖĮŠŲŪŽ
   * - luxembourgish
     - 110
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿äöüéëÄÖÜÉË
   * - malagasy
     - 94
     - 0123456789abdefghijklmnoprstvyzABDEFGHIJKLMNOPRSTVYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ôñÔÑ
   * - malay
     - 100
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - maltese
     - 104
     - 0123456789abdefghijklmnopqrstuvwxzABDEFGHIJKLMNOPQRSTUVWXZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ċġħżĊĠĦŻ
   * - montenegrin
     - 103
     - 0123456789abcdefghijklmnoprstuvzABCDEFGHIJKLMNOPRSTUVZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿čćšžźČĆŠŚŽŹ
   * - norwegian
     - 106
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿æøåÆØÅ
   * - polish
     - 118
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ąćęłńóśźżĄĆĘŁŃÓŚŹŻ
   * - portuguese
     - 128
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ
   * - romanian
     - 110
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ăâîșțĂÂÎȘȚ
   * - scottish_gaelic
     - 94
     - 0123456789abcdefghilmnoprstuABCDEFGHILMNOPRSTU!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿àèìòùÀÈÌÒÙ
   * - serbian_latin
     - 110
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿čćđžšČĆĐŽŠ
   * - slovak
     - 134
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ôäčďľňšťžáéíĺóŕúýÔÄČĎĽŇŠŤŽÁÉÍĹÓŔÚÝ
   * - slovene
     - 102
     - 0123456789abcdefghijklmnoprstuvzABCDEFGHIJKLMNOPRSTUVZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿čćđšžČĆĐŠŽ
   * - somali
     - 94
     - 0123456789abcdefghijklmnoqrstuwxyABCDEFGHIJKLMNOQRSTUWXY!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - spanish
     - 116
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áéíóúüñÁÉÍÓÚÜÑ¡¿
   * - swahili
     - 96
     - 0123456789abcdefghijklmnoprstuvwyzABCDEFGHIJKLMNOPRSTUVWYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - swedish
     - 106
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿åäöÅÄÖ
   * - tagalog
     - 95
     - 0123456789abdefghijklmnoprstuvyzABDEFGHIJKLMNOPRSTUVYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ñÑ₱
   * - turkish
     - 112
     - 0123456789abcdefghijklmnoprstuvyzABCDEFGHIJKLMNOPRSTUVYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿çğıöşüâîûÇĞİÖŞÜÂÎÛ
   * - uzbek_latin
     - 110
     - 0123456789abcdefghijklmnopqrstuvxyzABCDEFGHIJKLMNOPQRSTUVXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿çğɉñöşÇĞɈÑÖŞ
   * - vietnamese
     - 234
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áàảạãăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựíìỉĩịýỳỷỹỵÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴ
   * - welsh
     - 102
     - 0123456789abcdefghijlmnoprstuwyABCDEFGHIJLMNOPRSTUWY!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿âêîôŵŷÂÊÎÔŴŶ
   * - Zulu
     - 100
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - ancient_greek
     - 48
     - αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
   * - arabic
     - 101
     - ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيپچڢڤگ؟؛«»—0123456789٠١٢٣٤٥٦٧٨٩'ًٌٍَُِّْ'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
   * - arabic_diacritics
     - 2
     - 'ًٌٍَُِّْ'
   * - arabic_letters
     - 37
     - ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي
   * - arabic_punctuation
     - 5
     - ؟؛«»—
   * - persian_letters
     - 5
     - پچڢڤگ
   * - bangla
     - 70
     - অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃেৈোৌ্ৎংঃঁ০১২৩৪৫৬৭৮৯
   * - gujarati
     - 98
     - અઆઇઈઉઊઋએઐઓઔખગઘચછજઝઞટઠડઢણતથદધનપફબભમયરલવશસહળક્ષ૦૧૨૩૪૫૬૭૮૯!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~૰ઽ◌ંઃ॥ૐ઼ ઁ૱
   * - hindi
     - 68
     - अआइईउऊऋॠऌॡएऐओऔंःकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह०१२३४५६७८९।,?!:्ॐ॰॥
   * - hindi_digits
     - 10
     - ٠١٢٣٤٥٦٧٨٩
   * - generic_cyrillic_letters
     - 58
     - абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ
   * - russian
     - 109
     - абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯёыэЁЫЭъЪ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~₽
   * - ukrainian
     - 115
     - абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ґіїєҐІЇЄ₴
   * - multilingual
     - 205
     - english & french & german & italian & spanish & portuguese & czech & croatian & polish & dutch & norwegian & danish & finnish & swedish & §

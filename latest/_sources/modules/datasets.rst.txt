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
   * - czech
     - 130
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ
   * - danish
     - 106
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿æøåÆØÅ
   * - dutch
     - 114
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áéíóúüñÁÉÍÓÚÜÑ
   * - english
     - 100
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿
   * - finnish
     - 104
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿äöÄÖ
   * - german
     - 108
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿äöüßÄÖÜẞ
   * - french
     - 126
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ
   * - legacy_french
     - 123
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~°àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ£€¥¢฿
   * - hebrew
     - 235
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿אבגדהוזחטיךכלםמןנסעףפץצקרשתְֱֲֳִֵֶַָׇֹֺֻֽ־ֿ׀ׁׂ׃ׅׄ׆׳״֑֖֛֢֣֤֥֦֧֪֚֭֮֒֓֔֕֗֘֙֜֝֞֟֠֡֨֩֫֬֯ׯװױײיִﬞײַﬠﬡﬢﬣﬤﬥﬦﬧﬨ﬩שׁשׂשּׁשּׂאַאָאּבּגּדּהּוּזּטּיּךּכּלּמּנּסּףּפּצּקּרּשּתּוֹבֿכֿפֿﭏ₪
   * - italian
     - 120
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ
   * - latin
     - 94
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
   * - norwegian
     - 106
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿æøåÆØÅ
   * - polish
     - 118
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿ąćęłńóśźżĄĆĘŁŃÓŚŹŻ
   * - portuguese
     - 131
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áàâãéêëíïóôõúüçÁÀÂÃÉËÍÏÓÔÕÚÜÇ¡¿
   * - spanish
     - 116
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áéíóúüñÁÉÍÓÚÜÑ¡¿
   * - swedish
     - 106
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿åäöÅÄÖ
   * - vietnamese
     - 234
     - 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~°£€¥¢฿áàảạãăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựíìỉĩịýỳỷỹỵÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴ
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
     - 195
     - english & french & german & italian & spanish & portuguese & czech & polish & dutch & norwegian & danish & finnish & swedish & §

from langdetect import detect_langs

sentence = "This is a test sentence."
expected_lang = "en"
threshold_prob = 0.99


def test_get_lang():

    lang = detect_langs(sentence)[0]

    assert lang.lang == expected_lang
    assert lang.prob > threshold_prob

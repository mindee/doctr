from doctr.utils.lang_detect import get_language

sentence = "This is a test sentence."
expected_lang = "en"
threshold_prob = 0.92


def test_get_lang():

    lang = get_language(sentence)

    assert lang[0] == expected_lang
    assert lang[1] > threshold_prob

    lang = get_language("a")
    assert lang[0] == "unknown"
    assert lang[1] == 0.0

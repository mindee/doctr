from collections import Counter

from doctr.datasets import LATEX_TOKENS, VOCABS, tokenize_latex


def test_vocabs_duplicates():
    for key, vocab in VOCABS.items():
        assert isinstance(vocab, str)

        duplicates = [char for char, count in Counter(vocab).items() if count > 1]
        assert not duplicates, f"Duplicate characters in {key} vocab: {duplicates}"


def test_latex_tokens():
    assert isinstance(LATEX_TOKENS, list) and all(isinstance(token, str) for token in LATEX_TOKENS)
    # No duplicate, and every token is a single token for the LaTeX tokenizer
    assert len(LATEX_TOKENS) == len(set(LATEX_TOKENS))
    assert all(tokenize_latex(token) == [token] for token in LATEX_TOKENS)
    assert all(not any(char.isspace() for char in token) for token in LATEX_TOKENS)
    for token in ("a", "{", "^", "\\frac", "\\alpha", "\\operatorname*", "\\begin{pmatrix}", "\\end{array}", "\\\\"):
        assert token in LATEX_TOKENS
    # Tokens of the public formula datasets, beyond KaTeX
    for token in ("\\ensuremath", "\\textsc", "\\upmu", "\\-", "、", "θ", "数"):
        assert token in LATEX_TOKENS
    # The KaTeX tokens come first: their indices do not depend on the tokens found in the datasets
    assert LATEX_TOKENS.index("\\vcentcolon") < LATEX_TOKENS.index("\\ensuremath")

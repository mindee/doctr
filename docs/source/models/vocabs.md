# Vocabularies

A vocabulary is the set of characters a recognition model can predict. docTR ships more than 100 of them, covering
the main writing systems, in {data}`doctr.datasets.VOCABS`:

```python
from doctr.datasets import VOCABS

print(len(VOCABS["german"]))  # 108
print(VOCABS["german"][-8:])  # äöüßÄÖÜẞ
```

They are used to:

- **train** a recognition model on a language: `crnn_vgg16_bn(vocab=VOCABS["german"])`, or
  `--vocab german` with the [training script](../training/train.md#text-recognition);
- **restrict** a trained model to a subset of its characters with a
  [whitelist](../user_guide/ocr_predictor.md#restrict-the-output-characters).

:::{tip}
Vocabularies are plain strings, so you can combine them: `VOCABS["german"] + VOCABS["polish"]`. Remove the
duplicates while keeping the order with `"".join(dict.fromkeys(...))`.
:::

## Available vocabularies

Click a row to see all its characters. For Latin-script languages, the summary lists the characters added to the
`english` vocabulary.

```{include} _generated/vocabs.md
```

`VOCABS` also contains the building blocks these vocabularies are made of, such as `digits`, `punctuation`,
`currency` or the consonants and vowels of each Indic script.

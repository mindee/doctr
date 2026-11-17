# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Build a LaTeX token vocabulary for formula recognition: the default docTR tokens, extended with the tokens of
training labels which occur often enough (e.g. CJK characters in the formulas of Chinese documents)"""

import argparse
import json
from collections import Counter
from pathlib import Path

from doctr.datasets import LATEX_TOKENS, tokenize_latex


def main(args):
    counts: Counter[str] = Counter()
    for path in args.datasets:
        labels_path = Path(path).joinpath("labels.json") if Path(path).is_dir() else Path(path)
        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)
        for label in labels.values():
            counts.update(tokenize_latex(label))

    base = [] if args.from_scratch else list(LATEX_TOKENS)
    known = set(base)
    extra = [token for token, count in counts.most_common() if count >= args.min_count and token not in known]
    vocab = base + extra
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)

    rare = sum(1 for token, count in counts.items() if count < args.min_count and token not in known)
    print(f"{len(vocab)} tokens written to {args.output} ({len(extra)} added, {rare} rare tokens left out)")
    if extra:
        print(f"Added tokens: {extra[:50]}{' ...' if len(extra) > 50 else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a LaTeX token vocabulary from formula datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("datasets", type=str, nargs="+", help="dataset folders (with labels.json) or labels files")
    parser.add_argument("--output", type=str, default="vocab.json", help="path of the JSON vocab to write")
    parser.add_argument("--min-count", type=int, default=10, help="minimum number of occurrences of an added token")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="only keep the tokens of the datasets, instead of extending the default docTR tokens",
    )
    main(parser.parse_args())

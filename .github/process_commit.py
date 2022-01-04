"""
Borrowed & adapted from https://github.com/pytorch/vision/blob/main/.github/process_commit.py
This script finds the merger responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'. If there exists no PR associated with the commit or the PR is properly labeled,
this script is a no-op.
Note: we ping the merger only, not the reviewers, as the reviewers can sometimes be external to torchvision
with no labeling responsibility, so we don't want to bother them.
"""

import sys
from typing import Any, Optional, Set, Tuple

import requests

# For a PR to be properly labeled it should have one primary label and one secondary label

# Should specify the type of change
PRIMARY_LABELS = {
    "type: new feature",
    "type: bug",
    "type: enhancement",
    "type: misc",
}

# Should specify what has been modified
SECONDARY_LABELS = {
    "topic: documentation",
    "module: datasets",
    "module: io",
    "module: models",
    "module: transforms",
    "module: utils",
    "ext: api",
    "ext: demo",
    "ext: docs",
    "ext: notebooks",
    "ext: references",
    "ext: scripts",
    "ext: tests",
    "topic: build",
    "topic: ci",
    "topic: docker",
}

GH_ORG = 'mindee'
GH_REPO = 'doctr'


def query_repo(cmd: str, *, accept) -> Any:
    response = requests.get(f"https://api.github.com/repos/{GH_ORG}/{GH_REPO}/{cmd}", headers=dict(Accept=accept))
    return response.json()


def get_pr_number(commit_hash: str) -> Optional[int]:
    # See https://docs.github.com/en/rest/reference/repos#list-pull-requests-associated-with-a-commit
    data = query_repo(f"commits/{commit_hash}/pulls", accept="application/vnd.github.groot-preview+json")
    if not data:
        return None
    return data[0]["number"]


def get_pr_merger_and_labels(pr_number: int) -> Tuple[str, Set[str]]:
    # See https://docs.github.com/en/rest/reference/pulls#get-a-pull-request
    data = query_repo(f"pulls/{pr_number}", accept="application/vnd.github.v3+json")
    merger = data["merged_by"]["login"]
    labels = {label["name"] for label in data["labels"]}
    return merger, labels


def main(args):
    pr_number = get_pr_number(args.commit_hash)
    if not pr_number:
        sys.exit(0)

    merger, labels = get_pr_merger_and_labels(pr_number)
    is_properly_labeled = bool(PRIMARY_LABELS.intersection(labels) and SECONDARY_LABELS.intersection(labels))
    if not is_properly_labeled:
        print(f"@{merger}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PR label checker',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('commit_hash', type=str, help='Hash of the commit')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

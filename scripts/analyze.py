# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import matplotlib.pyplot as plt
from doctr.models import zoo
from doctr.documents import read_pdf
from doctr.utils.visualization import visualize_page

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(args):

    if args.model not in zoo.__all__:
        raise ValueError('only the following end-to-end predictors are supported:', zoo.__all__)

    model = models.__dict__[args.model](pretrained=True)

    doc = read_pdf(args.path)

    out = model([doc])

    for page, img in zip(out[0].pages, doc):
        visualize_page(page.export(), img)
        plt.show(block=True)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', type=str, help='Path to the input PDF document')
    parser.add_argument('--model', type=str, default='ocr_db_crnn', help='OCR model to use for analysis')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

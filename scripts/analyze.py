# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import matplotlib.pyplot as plt
from doctr.models import ocr_predictor
from doctr.documents import read_pdf
from doctr.utils.visualization import visualize_page


def main(args):

    model = ocr_predictor(args.detection, args.recognition, pretrained=True)

    doc = read_pdf(args.path)

    out = model(doc)

    for page, img in zip(out.pages, doc):
        visualize_page(page.export(), img)
        plt.show(block=not args.noblock)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', type=str, help='Path to the input PDF document')
    parser.add_argument('--detection', type=str, default='db_resnet50',
                        help='Text detection model to use for analysis')
    parser.add_argument('--recognition', type=str, default='crnn_vgg16_bn',
                        help='Text recognition model to use for analysis')
    parser.add_argument("--noblock", dest="noblock", help="Disables blocking visualization", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

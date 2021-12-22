# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import os

os.environ['USE_TORCH'] = '1'

import argparse
import logging

import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch

from doctr.models import obj_detection


def inference(artefact, root_dir, test_list, visualize):

    '''
    {"geometry": <*,4>,"class": <*>,"codes": <*>,"score": <*>}
    qr_code = 1, bar_code = 2, logo = 3, photo = 4
    '''
    cm = {'1': [(0, 0, 150)], '2': [(0, 0, 0)], '3': [(0, 150, 0)], '4': [(150, 0, 0)]}
    cl_map = {'1': "QR_Code", "2": "Bar_Code", "3": "Logo", "4": "Photo"}
    inf_dic = {}
    bbox = []
    art_class = []
    c_score = []
    for val in test_list:
        im_read = cv2.imread(os.path.join(root_dir, val))
        imm = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
        imm2 = imm / 255
        if torch.cuda.is_available():
            imm2 = [torch.tensor(imm2, dtype=torch.float32).permute(2, 0, 1).cuda()]
        else:
            imm2 = [torch.tensor(imm2, dtype=torch.float32).permute(2, 0, 1)]
        pred = artefact(imm2)
        tg1 = pred[0]['labels'].detach().cpu().numpy()
        tg = pred[0]['boxes'].detach().cpu().numpy()
        tg2 = pred[0]['scores'].detach().cpu().numpy()
        tg = [list(i) for i in tg]
        bbox.append(tg)
        art_class.append(tg1)
        c_score.append(tg2)
        for ind_2, val_2 in enumerate(tg):
            cv2.rectangle(imm, (int(val_2[0]), int(val_2[1])), (int(val_2[2]), int(val_2[3])),
                          cm[str(int(tg1[ind_2]))][0], 2)
            text_size, _ = cv2.getTextSize(cl_map[str(int(tg1[ind_2]))], cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            text_w, text_h = text_size
            cv2.rectangle(imm, (int(val_2[0]), int(val_2[1])), (int(val_2[0]) + text_w, int(val_2[1]) - text_h),
                          cm[str(int(tg1[ind_2]))][0], -1)
            cv2.putText(imm, cl_map[str(int(tg1[ind_2]))], (int(val_2[0]), int(val_2[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        if visualize:
            figure(figsize=(9, 7), dpi=100)
            plt.imshow(imm)
            plt.show()
        inf_dic.update({"geometry": str(bbox), "class": str(art_class), "score": str(c_score)})

    return inf_dic


def main(args):
    print(args)

    model = obj_detection.__dict__[args.arch](pretrained=True, num_classes=5)
    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        logging.warning("No accessible GPU, target device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()
    model.eval()
    inf_dic = inference(model, args.root_dir, os.listdir(args.root_dir), args.visualize)
    print(inf_dic)


def parse_args():
    parser = argparse.ArgumentParser(description="Artefact Detection Training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('arch', type=str, help='text-detection model to train')
    parser.add_argument('root_dir', type=str, help='path to image folder')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import os
from itertools import chain
from typing import List

os.environ['USE_TORCH'] = '1'

import argparse
import json
import logging

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pyzbar.pyzbar import decode
from pyzxing import BarCodeReader
from torchvision.ops import MultiScaleRoIAlign

from doctr.utils.metrics import LocalizationConfusion


def pre_filter(scores: List, labels: List, ):
    '''
    removes repeating labels i.e in a cluster multiples boxes with same labels
    Example:
        >>> s = [0.9, 0.7, 0.6, 0.65, 0.43, 0.78, 0.23, 0.976, 0.121]
        >>> l = [1, 1, 4, 3, 3, 2, 2, 2, 5]
        >>> s=  [4, 5, 1, 2, 3]
        >>> l= [0.6, 0.121, 0.9, 0.976, 0.65]
    Args:
        s: list of scores
        l: list labels
    '''
    ioi = [i for i, j in enumerate(labels) if labels.count(j) > 1]  # indices of al repeating elements
    sc = [j for i, j in enumerate(scores) if i not in ioi]  # list of score excluding all repeated elements
    g = list(set([j for i, j in enumerate(labels) if labels.count(j) > 1]))  # gives repeating elements
    lc = [i for i in labels if i not in g]
    jo = [duplicates(labels, q) for q in
          g]  # returns [ [],[],[] ] where sub-arrays repeating labels of individual classes
    for ind, val in enumerate(jo):
        temp_dic = {}
        for j in val:  # indices of the repeated elements
            temp_dic.update({scores[j]: j})
        # at a position to choose among logo or a code/take into account relative area as well.
        # take relative intensity into account as well
        s_back, l_back = list(sorted(temp_dic.items(), key=lambda t: t[0]))[-1]
        sc.append(s_back)
        lc.append(labels[l_back])
    return photo_id_nms(sc, lc)


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def photo_id_nms(scores: List, labels: List):
    '''
    Removes photo label if found overlapping with other labels irrespective of the scores
    Args:
        s: list of scores
        l:list of labels
    '''
    t = dict(zip(scores, labels))
    ni = {}
    for i, j in t.items():
        if j != 4:  # label of photo
            ni.update({i: j})
    scores = list(ni.keys())
    labels = list(ni.values())
    return scores, labels


def logo_thresh(bboxes: List, scores: List, labels: List, img: torch.Tensor):
    '''
    Removes predictions of logo labels if thresholds are not met. Prediction score head of faster rcnn
    and location along height of the document in combination sets a threshold for the logo labels.
    Args:
         bboxs:List of bounding boxes
         s:List of scores
         l:List of labels
         img: image
    '''
    ind = [i for i, j in enumerate(labels) if
           (j == 3 and scores[i] < 0.3 and bboxes[i][1] > 0.45 * img[0].cpu().numpy().shape[
               1])]  # prune based on thresholds
    bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in ind]
    scores = np.delete(np.array(scores), ind)
    labels = np.delete(np.array(labels), ind)
    return list(bboxes), list(scores), list(labels)


def post_nms(pred: List[dict], img: torch.tensor):
    c = LocalizationConfusion(iou_thresh=0.5, )
    pr_label = pred[0]['labels'].detach().cpu().numpy()  # prediction boxes
    single_b = []
    single_l = []
    single_s = []
    pr_box = pred[0]['boxes'].detach().cpu().numpy()
    pr_score = pred[0]['scores'].detach().cpu().numpy()
    ignore_list = []
    for i in range(len(pr_label)):
        cluster_box = []
        cluster_label = []
        cluster_score = []
        if pr_score[i] not in ignore_list:
            temp_b = []
            temp_l = []
            temp_s = []
            for j in range(i + 1, len(pr_label)):
                c.reset()
                c.update(np.asarray([list(pr_box[i])]), np.asarray(list([pr_box[j]])))
                if c.summary()[2] >= 0.3:  # iou threshold
                    ignore_list.append(pr_score[j])
                    temp_b.append(pr_box[j])
                    temp_l.append(pr_label[j])
                    temp_s.append(pr_score[j])
            if temp_b != []:
                cluster_score.append(list(chain.from_iterable([[pr_score[i]], temp_s])))
                cluster_label.append(list(chain.from_iterable([[pr_label[i]], temp_l])))
                cluster_box.append(list(chain.from_iterable([[pr_box[i]], temp_b])))
                scores, labels = pre_filter(cluster_score[0], cluster_label[0])  # at this stage max 2
                if pr_box[list(pr_score).index(scores[0])][1] < 0.3 * img[0].cpu().numpy().shape[1]:
                    tpp = dict(zip(scores, labels))
                    s_nms, l_nms = list(sorted(tpp.items(), key=lambda i: i[0]))[-1]
                    single_s.append(s_nms)
                    single_b.append(pr_box[list(pr_score).index(s_nms)])
                    single_l.append(l_nms)
                else:
                    tpp = dict(zip(scores, labels))
                    s_nms, l_nms = list(sorted(tpp.items(), key=lambda i: i[0]))[-1]
                    if l_nms == 3 and s_nms > 0.3:
                        single_s.append(s_nms)
                        single_b.append(pr_box[list(pr_score).index(s_nms)])
                        single_l.append(l_nms)
                    elif l_nms == 1 or l_nms == 2:
                        single_s.append(s_nms)
                        single_b.append(pr_box[list(pr_score).index(s_nms)])
                        single_l.append(l_nms)
                    else:
                        pass
            else:
                single_b.append(pr_box[i])
                single_l.append(pr_label[i])
                single_s.append(pr_score[i])
    return logo_thresh(single_b, single_s, single_l, img)


def inference_script(artefact, root_dir, test_list, code_KIE, save_image, save_to_folder):
    '''
    {"geometry": <*,4>,"class": <*>,"KIE": <*>,"score": <*>}
    qr_code = 1, bar_code = 2, logo = 3, photo = 4
    '''
    reader = BarCodeReader()
    inf_dic = {}
    bbox = []
    art_class = []
    code_info = []
    c_score = []
    for ind, val in enumerate(test_list):
        im_read = cv2.imread(os.path.join(root_dir, val))
        kold = im_read.copy()
        kold = cv2.cvtColor(kold, cv2.COLOR_BGR2GRAY)
        imm = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
        imm = imm / 255
        imm2 = [torch.tensor(imm, dtype=torch.float32).permute(2, 0, 1)]
        pred = artefact(imm2)
        tg, tg2, tg1 = post_nms(pred, imm2)  # box, score, label
        tg = [list(i) for i in tg]
        bbox.append(tg)
        art_class.append(tg1)
        c_score.append(tg2)
        if save_image:
            for ind_2, val_2 in enumerate(tg):
                ji = {'1': [(0, 150, 150)], '2': [(0, 0, 0)], '3': [(0, 150, 0)], '4': [(150, 0, 0)]}
                juj = {'1': "QR_Code", "2": "Bar_Code", "3": "Logo", "4": "Photo"}
                cv2.rectangle(imm, (int(val_2[0]), int(val_2[1])), (int(val_2[2]), int(val_2[3])),
                              ji[str(int(tg1[ind_2]))][0], 2)
                imm = cv2.putText(imm, str(tg2[ind_2]), (int(val_2[0]), int(val_2[3]) + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                  1.5,
                                  ji[str(int(tg1[ind_2]))][0], 2, )
                imm = cv2.putText(imm, juj[str(int(tg1[ind_2]))], (int(val_2[0]), int(val_2[1])),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  1.5,
                                  ji[str(int(tg1[ind_2]))][0], 2, )
            theimagee = Image.fromarray((imm * 255).astype(np.uint8))
            theimagee.save(f'{save_to_folder}/{val}')
        if code_KIE:
            for i in range(len(tg)):
                txt = []
                if int(tg1[i]) == 1:
                    pio = kold[int(tg[i][1]) - 6:int(tg[i][3]) + 6, int(tg[i][0]) - 6:int(tg[i][2]) + 6]
                    try:
                        txt = reader.decode_array(pio)
                        if "parsed" in list(txt[0].keys()):
                            code_info.append(txt[0]["parsed"].decode())
                        else:
                            code_info.append(["Unreadable/Not a code"])
                    except ZeroDivisionError:
                        code_info.append(["Unreadable/Not a code"])

                elif int(tg1[i]) == 2:
                    pio = kold[int(tg[i][1]) + 10:int(tg[i][3]) - 5, int(tg[i][0]) - 2:int(tg[i][2]) + 2]
                    try:
                        txt = decode(pio)
                        code_info.append(txt)
                    except KeyError:
                        code_info.append(["Unreadable/Not a code"])

            inf_dic.update(
                {"geometry": str(bbox), "class": str(art_class), "score": str(c_score), "KIE": str(code_info)})
        else:
            inf_dic.update({"geometry": str(bbox), "class": str(art_class), "score": str(c_score)})
    return inf_dic


def main(args):
    torch.backends.cudnn.benchmark = True

    # Filter keys
    state_dict = {
        k: v for k, v in torchvision.models.detection.__dict__[args.arch](pretrained=True).state_dict().items()
        if not k.startswith('roi_heads.')
    }
    defaults = {"min_size": 800, "max_size": 1300,
                "box_fg_iou_thresh": 0.5,
                "box_bg_iou_thresh": 0.5,
                "box_detections_per_img": 150, "box_score_thresh": 0.15, "box_positive_fraction": 0.35,
                "box_nms_thresh": 0.2,
                "rpn_pre_nms_top_n_train": 2000, "rpn_pre_nms_top_n_test": 1000,
                "rpn_post_nms_top_n_train": 2000, "rpn_post_nms_top_n_test": 1000,
                "rpn_nms_thresh": 0.2,
                "rpn_batch_size_per_image": 250
                }
    kwargs = {**defaults}

    model = torchvision.models.detection.__dict__[args.arch](pretrained=False, num_classes=5, **kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.roi_heads.box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7),
                                                      sampling_ratio=2)
    anchor_sizes = ((16), (64), (128), (264))
    aspect_ratios = ((0.5, 1.0, 2.0, 3.0,)) * len(anchor_sizes)
    model.rpn.anchor_generator.sizes = anchor_sizes
    model.rpn.anchor_generator.aspect_ratios = aspect_ratios
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
    root_dir = args.root_dir
    save_to_folder = args.save_to
    checkpoint_path = args.checkpoint_path
    save_image = args.plot
    code_KIE = args.KIE
    save_j = args.save_json
    if_save = args.json
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    test_list = os.listdir(args.root_dir)
    inf_dic = inference_script(model, root_dir, test_list, code_KIE,
                               save_image, save_to_folder)
    if if_save:
        with open(save_j, 'w', encoding='utf-8') as f:
            json.dump(inf_dic, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Artefact Detection Training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('root_dir', type=str, help='path to image folder')
    parser.add_argument('checkpoint_path', type=str, help='path to torch checkpoint')
    parser.add_argument('-save_to', type=str, required=False,
                        help="Use this along with --plot True to save images to the desired location")
    parser.add_argument('-save_json', type=str, required=False,
                        help="Use this along with --json True")
    parser.add_argument('--json', nargs='?', const=False, type=bool,
                        help="Optional: If user wants to save json output")
    parser.add_argument('--KIE', nargs='?', const=False, type=bool,
                        help="Optional:If user wants to extract information from codes")
    parser.add_argument('--plot', nargs="?", const=False, type=bool,
                        help="Optional:If the user wants to plot and save the predictions")
    args = parser.parse_args()
    if args.plot and (args.save_to is None):
        parser.error(
            "User has requested to plot the predictions on the images but have not provided the destination path."
            " Please provide the path to save the images. Usage: --plot True -save_to <str(path)>")
    if args.json and (args.save_json is None):
        parser.error(
            "User has requested to save the predictions but have not provided the destination path. Please provide the"
            "path to save the images. Usage: --json True -save_json <str(path)>")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

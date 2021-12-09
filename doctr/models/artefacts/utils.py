# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from itertools import chain
from typing import List

import cv2
import doctr
import numpy as np
import torch
from PIL import Image
from pyzbar.pyzbar import decode
from pyzxing import BarCodeReader


def aspect_ratio(x: np.array, area):
    '''
    :param area:
    :param x: bboxes
    :return: aspect ratio and relative area of the bbox
    '''
    x_ = x[2] - x[0]
    y_ = x[3] - x[1]
    if y_ > x_:
        return y_ / x_, (y_ * x_) / (area.shape[0] * area.shape[1])
    else:
        return x_ / y_, (y_ * x_) / (area.shape[0] * area.shape[1])


# def xtreme_ai(b, s, l, img: torch.Tensor):
#     '''takes into account extreme aspect ratio,relative area and other such statistical numbers'''
#     # area =...
#     return None  # still to be engineered


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
    for i in range(len(jo)):
        temp_dic = {}
        for j in jo[i]:  # indices of the repeated elements
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
        if j != 4:
            ni.update({i: j})
    scores = list(ni.keys())
    labels = list(ni.values())
    return scores, labels


def logo_thresh(bboxs: List, scores: List, labels: List, img: torch.Tensor, xtreme_ai: bool):
    '''
    Removes predictions of logo labels if thresholds are not met
    Args:
         b:List of bounding boxes
         s:List of scores
         l:List of labels
         img: image
    '''
    ind = [i for i, j in enumerate(labels) if
           (j == 3 and scores[i] < 0.3 and bboxs[i][1] > 0.5 * img[0].cpu().numpy().shape[
               1])]  # prune based on thresholds
    bboxs = [bboxs[i] for i in range(len(bboxs)) if i not in ind]
    scores = np.delete(np.array(scores), ind)
    labels = np.delete(np.array(labels), ind)
    if xtreme_ai:
        return xtreme_ai(bboxs, scores, labels, img)
    else:
        return list(bboxs), list(scores), list(labels)


def post_nms(pop, img: torch.tensor, xtreme_ai=False):
    c = doctr.utils.metrics.LocalizationConfusion(iou_thresh=0.5, )
    pr_label = pop[0]['labels'].detach().cpu().numpy()  # prediction boxes
    single_b = []
    single_l = []
    single_s = []
    pr_box = pop[0]['boxes'].detach().cpu().numpy()
    pr_score = pop[0]['scores'].detach().cpu().numpy()
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
    return logo_thresh(single_b, single_s, single_l, img, xtreme_ai)


def inference_script(artefact, root_dir, test_list, code_KIE, save_image, save_to_folder):
    '''
    {"geometry":<*,4>,"class":<*>,"KIE":<*>,"score":<*>}
    qr_code = 1,bar_code = 2,photo = 3,logo = 4
    '''
    reader = BarCodeReader()
    inf_dic = {}
    bbox = []
    art_class = []
    code_info = []
    c_score = []
    for j in range(len(test_list)):
        im_read = cv2.imread(os.path.join(root_dir, test_list[j]))
        kold = im_read.copy()
        kold = cv2.cvtColor(kold, cv2.COLOR_BGR2GRAY)
        imm = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
        imm = imm / 255
        imm2 = [torch.tensor(imm, dtype=torch.float32).permute(2, 0, 1)]
        pop = artefact(imm2)
        tg, tg2, tg1 = post_nms(pop, imm2)  # box, score, label
        tg = [list(i) for i in tg]
        bbox.append(tg)
        art_class.append(tg1)
        c_score.append(tg2)
        if save_image:
            for i in range(len(tg)):
                ji = {'1': [(0, 150, 150)], '2': [(0, 0, 0)], '3': [(0, 150, 0)], '4': [(150, 0, 0)]}
                juj = {'1': "QR_Code", "2": "Bar_Code", "3": "Logo", "4": "Photo"}
                cv2.rectangle(imm, (int(tg[i][0]), int(tg[i][1])), (int(tg[i][2]), int(tg[i][3])),
                              ji[str(int(tg1[i]))][0], 2)
                imm = cv2.putText(imm, str(tg2[i]), (int(tg[i][0]), int(tg[i][3]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                  ji[str(int(tg1[i]))][0], 2, )
                imm = cv2.putText(imm, juj[str(int(tg1[i]))], (int(tg[i][0]), int(tg[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                  1.5,
                                  ji[str(int(tg1[i]))][0], 2, )
            theimagee = Image.fromarray((imm * 255).astype(np.uint8))
            ll = test_list[j]
            theimagee.save(f'{save_to_folder}/{ll}')
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

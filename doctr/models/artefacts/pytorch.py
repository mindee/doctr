# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import os

os.environ['USE_TORCH'] = '1'

import argparse
import json
import logging
import torch
import torchvision
from torchvision.ops import MultiScaleRoIAlign

from doctr.models.artefacts.utils import inference_script


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

import json
import os
from pathlib import Path
from typing import Any
from itertools import zip_longest
import numpy as np
from tqdm import tqdm

from .datasets import AbstractDataset
from .utils import convert_target_to_relative, crop_bboxes_from_image

__all__ = ["COCO"]


class COCO(AbstractDataset):
    """
    we will do it afterwards

    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        train: bool = True,
        use_polygons: bool = False,
        recognition_task: bool = False,
        detection_task: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            img_folder, pre_transforms=convert_target_to_relative if not recognition_task else None, **kwargs
        )
        # checking task
        if recognition_task and detection_task:
            raise ValueError(
                " 'recognition' and 'detection task' cannot be set to True simultaneously. "
                + " To get the whole dataset with boxes and labels leave both parameters to False "
            )

        if not os.path.exists(label_path) or not os.path.exists(img_folder):
            raise FileNotFoundError(f"unable to find {label_path if not os.path.exists(label_path) else img_folder}")

        # loading annotations
        with open(label_path, 'r') as f:
            coco_data = json.load(f)
        
        self.train = train
        self.data: list[tuple[str | Path | np.ndarray, str | dict[str, Any] | np.ndarray]] = []

      
        # processing annotations
        for img_id, img_info in  tqdm(coco_data['imgs'].items(), desc = "Loading COCO-text"):
            img_path = os.path.join(img_folder, img_info['file_name'])
            annots = [a for a in coco_data['anns'].values() if a['image_id'] == int(img_id) and a['legibility'] == "legible"]
            
            text_targets, box_targets = [], []
            for annot in annots:
                x, y, w, h = annot['bbox']
                box = [x, y, x+w, y+h]
                text_targets.append(annot.get("utf8_string", ""))
                box_targets.append(box)
                
            if recognition_task:
                crops = crop_bboxes_from_image(img_path, np.array(box_targets, dtype = int))
                for crop, label in zip_longest(crops, text_targets, fillvalue="MISSING"):
                    if label:
                        self.data.append((crop, label))
            
            elif detection_task:
                self.data.append((img_path, np.array(box_targets, dtype = int)))
                
                
            else:
                self.data.append((img_path, {"boxes": np.array(box_targets, dtype = int), "labels": text_targets}))
                
        self.root = img_folder
        
    def extra_repr(self):
        return super().extra_repr()
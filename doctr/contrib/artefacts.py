import math
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
import torchvision

from doctr.models.preprocessor import PreProcessor
from doctr.utils.data import download_from_url

__all__ = ["ArtefactDetector"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "yolov8_artefact": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "labels": ["bar_code", "qr_code", "logo", "photo"],
        "url": "https://github.com/mindee/doctr/releases/download/v0.8.1/yolov8_artefact-6f78d9e9.onnx",
    },
}


class ArtefactDetector:
    """
    A class to detect artefacts in images

    Args:
    ----
        arch: the architecture to use
        batch_size: the batch size to use
        model_path: the path to the model to use
        labels: the labels to use
        mask_labels: the mask labels to use
        conf_threshold: the confidence threshold to use
        iou_threshold: the intersection over union threshold to use
        **kwargs: additional arguments to be passed to `doctr.models.preprocessor.PreProcessor`
    """

    def __init__(
        self,
        arch: str = "yolov8_artefact",
        batch_size: int = 2,
        model_path: Optional[str] = None,
        labels: Optional[List[str]] = default_cfgs["yolov8_artefact"]["labels"],
        mask_labels: Optional[List[str]] = None,
        conf_threshold: float = 0.9,
        iou_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.onnx_model = self._init_model(default_cfgs[arch]["url"], model_path)
        self.labels = labels or default_cfgs[arch]["labels"]
        self.input_shape = default_cfgs[arch]["input_shape"]
        self.mask_labels = mask_labels
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = ort.InferenceSession(
            self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.pre_processor = PreProcessor(
            output_size=default_cfgs[arch]["input_shape"][-2:],
            batch_size=batch_size,
            mean=default_cfgs[arch]["mean"],
            std=default_cfgs[arch]["std"],
            **kwargs,
        )

    def _init_model(self, url: str, model_path: Optional[str] = None, **kwargs: Any):
        return model_path if model_path else download_from_url(url, cache_subdir="models", **kwargs)

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).
            ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                calculated based on the size difference between the two images.
            padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
                rescaling.
            xywh (bool): The box format is xywh or not, default=False.

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])
        return boxes

    def regularize_rboxes(self, rboxes):
        """
        Regularize rotated boxes in range [0, pi/2].

        Args:
            rboxes (np.ndarray): (N, 5), xywhr.

        Returns:
            (np.ndarray): The regularized boxes.
        """
        x, y, w, h, t = np.split(rboxes, 5, axis=-1)
        # Swap edge and angle if h >= w
        w_ = np.where(w > h, w, h)
        h_ = np.where(w > h, h, w)
        t = (t + np.where(w > h, 0, math.pi / 2)) % math.pi
        return np.concatenate((x, y, w_, h_, t), axis=-1)  # regularized boxes

    def _get_covariance_matrix(self, boxes):
        """
        Generating covariance matrix from obbs.

        Args:
            boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

        Returns:
            (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
        """
        # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
        gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
        a, b, c = gbbs.split(1, dim=-1)
        cos = c.cos()
        sin = c.sin()
        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    def batch_probiou(self, obb1, obb2, eps=1e-7):
        """
        Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

        Args:
            obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
            obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
        """
        obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
        obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

        x1, y1 = obb1[..., :2].split(1, dim=-1)
        x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = (x.squeeze(-1)[None] for x in self._get_covariance_matrix(obb2))

        t1 = (
            ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
            / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        ) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
        t3 = (
            ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
            / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
            + eps
        ).log() * 0.5
        bd = (t1 + t2 + t3).clamp(eps, 100.0)
        hd = (1.0 - (-bd).exp() + eps).sqrt()
        return 1 - hd

    def xywh2xyxy(self, x):
        """
        Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

        Args:
            x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

        Returns:
            y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
        """
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
        return y

    def nms_rotated(self, boxes, scores, threshold=0.45):
        """
        NMS for obbs, powered by probiou and fast-nms.

        Args:
            boxes (torch.Tensor): (N, 5), xywhr.
            scores (torch.Tensor): (N, ).
            threshold (float): IoU threshold.

        Returns:
        """
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int8)
        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = self.batch_probiou(boxes, boxes).triu_(diagonal=1)
        pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
        return sorted_idx[pick]

    def non_max_suppression(
        self,
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
    ):
        """
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Args:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            multi_label (bool): If True, each box may have multiple labels.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels.
            in_place (bool): If True, the input prediction tensor will be modified in place.

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """
        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
        if isinstance(
            prediction, (list, tuple)
        ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        if not rotated:
            if in_place:
                prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy
            else:
                prediction = torch.cat(
                    (self.xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1
                )  # xywh to xyxy

        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]) and not rotated:
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = self.xywh2xyxy(lb[:, 1:5])  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = torch.where(cls > conf_thres)
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            # if classes is not None:
            #    x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores
            if rotated:
                boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
                i = self.nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c  # boxes (offset by class)
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            # # Experimental
            # merge = False  # use merge-NMS
            # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            #     from .metrics import box_iou
            #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
            #     weights = iou * scores[None]  # box weights
            #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            #     redundant = True  # require redundant detections
            #     if redundant:
            #         i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output

    def xywhr2xyxyxyxy(self, rboxes):
        """
        Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
        be in degrees from 0 to 90.

        Args:
            rboxes (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

        Returns:
            (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
        """
        is_numpy = isinstance(rboxes, np.ndarray)
        cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

        ctr = rboxes[..., :2]
        w, h, angle = (rboxes[..., i : i + 1] for i in range(2, 5))
        cos_value, sin_value = cos(angle), sin(angle)
        vec1 = [w / 2 * cos_value, w / 2 * sin_value]
        vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
        vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
        vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)

    def xyxyxyxyn(self, output, org_shape):
        """Return the boxes in xyxyxyxy format, (N, 4, 2)."""
        xyxyxyxyn = self.xywhr2xyxyxyxy(output)
        xyxyxyxyn[..., 0] /= org_shape[1]
        xyxyxyxyn[..., 1] /= org_shape[0]
        return xyxyxyxyn

    def _print_res(self, output, org_shape):
        c, conf = int(output[:, -1]), float(output[:, -2])
        line = (c, *(self.xyxyxyxyn(output, org_shape).view(-1)))
        print(f"{line[0]:>3} {conf:.2f} {' '.join(f'{x:.4f}' for x in line[1:])}")

    def _postprocess(self, output: np.ndarray, input_images: np.ndarray):
        # output to tensor
        output = torch.from_numpy(output)

        preds = self.non_max_suppression(
            output,
            self.conf_threshold,
            self.iou_threshold,
            agnostic=False,
            max_det=30,
            nc=len(self.labels),
            classes=self.labels,
            rotated=True,
        )

        results = []
        for pred, orig_img in zip(preds, input_images):
            print(self.input_shape[-2:])
            print(orig_img.shape[:-1])
            rboxes = self.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            print(rboxes)
            rboxes[:, :4] = self.scale_boxes(self.input_shape[-2:], rboxes[:, :4], orig_img.shape[:-1], xywh=True)
            # xywh, r, conf, cls
            rboxes = torch.from_numpy(rboxes).to(dtype=torch.int32)
            print(rboxes)
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)

        return results

    def __call__(self, inputs: List[np.ndarray]) -> Any:
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        processed_batches = [batch.numpy() for batch in self.pre_processor(inputs)]
        outputs = [self.session.run(None, {model_inputs[0].name: batch}) for batch in processed_batches]
        outputs = np.concatenate([output[0] for output in outputs], axis=0)
        return self._postprocess(outputs, inputs)

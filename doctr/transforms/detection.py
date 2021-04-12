# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import random
import cv2
import albumentations as A
from typing import Tuple


__all__ = ['geometric_transform', 'non_geometric_transform']


def geometric_transform(
    image: np.array,
    mask: np.array
) -> Tuple[np.array, np.array]:
    """Applies a pipeline of geometric transformation to an image the corresponding mask.
​
​    Args:
        image : np.array, input image
        mask : np.array, mask of the input image
​
    Returns:
        A tuple of transformed image and mask

    """

    # DISTORTION
    distortion = []
    # Randomly apply perspective to the input image
    # distortion.append(A.Perspective(scale=(0.05, 0.1), p=0.5, mask_pad_val=0))
    # Randomly  apply elastic transformation (Simard 2003)
    distortion.append(
        A.ElasticTransform(
            p=.5, alpha=12, sigma=50, alpha_affine=12, border_mode=cv2.BORDER_CONSTANT, mask_value=0
        )
    )
    distortion = A.OneOf(distortion, p=.5)

    inter_styles = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
    inter = random.choice(inter_styles)

    # ROTATION, SCALE, SHIFT
    # Randomly apply affine transforms: translate, scale and rotate the input.
    rotation = A.ShiftScaleRotate(
        p=0.25, scale_limit=.25, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, mask_value=0, interpolation=inter
    )

    # RESIZE
    h, w, _ = image.shape
    h_ratio, w_ratio = (np.random.uniform(0.85, 1.15), np.random.uniform(0.85, 1.15))
    # Randomly change the aspect ratio
    resize = A.Resize(height=int(h_ratio * h), width=int(w_ratio * w), p=0.5, interpolation=inter)

    # Compose distortion + rotation/shift/scale + resize and apply to image and mask
    aug = A.Compose([distortion, rotation, resize], p=.7)
    augmented = aug(image=image, mask=mask)
    image = augmented["image"]
    mask = augmented["mask"]

    return image, mask


def non_geometric_transform(
    image: np.array,
) -> np.array:
    """Applies a pipeline of non geometric transformation to an image.
    Transformations are pixel-wise, and thus can be applied to image only.
​
​    Args:
        image : np.array, input image
​
    Returns:
        image: np.array, augmented image
    """

    # COLOR
    color = []
    # Randomly shift each channel to recolorize image
    color.append(A.RGBShift(p=.5, r_shift_limit=150, g_shift_limit=150, b_shift_limit=150))
    color.append(A.RandomGamma(p=.5))  # Randomly power the image (default between 0.8 and 1.2)
    color.append(A.ToGray(p=.5))  # Randomly Convert to grayscale
    color.append(A.CLAHE(p=.5))  # Randomly apply Contrast Limited Adaptive Histogram Equalization
    color = A.OneOf(color, p=.5)  # Randomly choose one of the color transformations

    # EXPOSURE
    exposure = []
    exposure.append(A.RandomBrightnessContrast(p=.5))  # Randomly change brightness and constrast
    exposure.append(A.RandomShadow(shadow_roi=(0, 0, 1, 1), shadow_dimension=3, p=.5))  # Randomly shadow a part of the image
    exposure.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=70, p=.5))  # Randomly add a sunflare on the image
    exposure = A.OneOf(exposure, p=.3)  # Randomly choose one of the exposure transformations

    # BLUR
    blur = []
    blur.append(A.GaussianBlur(blur_limit=3, p=.5))  # Randomly blur the image with a gaussian filter
    blur.append(A.MotionBlur(blur_limit=3, p=.5))  # Randomly blur the image with a motion style
    blur = A.OneOf(blur, p=.25)  # Randomly choose one of the blurring transformations

    # NOISE
    noise = []
    noise.append(A.GaussNoise(p=.5))  # Randomly applies gaussian noise
    noise.append(A.ISONoise(p=.5, intensity=[.1, 1.]))  # Randomly applies camera sensor noise
    noise = A.OneOf(noise, p=.5)  # Randomly choose one of the noise transformations

    # compose blur + color + exposure + noise and apply to image
    aug = A.Compose([blur, color, exposure, noise], p=1)
    augmented = aug(image=image)
    image = augmented["image"]

    return image

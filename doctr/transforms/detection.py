import numpy as np
import random
import imgaug.augmenters as iaa
import cv2
from ocr_tools.degraders import degraders
import albumentations as A
from ocr_tools.degraders.iaa_augmentations import augment_image_and_mask
​
​
def geometric_pipeline(
    image: np.array,
    mask: np.array
    ) -> Tuple[np.array, np.array]:
    """Applies a pipeline of geometric transformation to an image and it mask
​
​    Args:
        image : np.array, input image
        mask : np.array, mask of the input image
​
    Returns:
        A tuple of transformed image and mask
    """

    geometric_augmentation = []
​
    # # distortions
    distortions = [A.OpticalDistortion(p=1.0, distort_limit=0.05, shift_limit=0.05),
                   A.ElasticTransform(p=1.0, alpha=12, sigma=50, alpha_affine=12, border_mode=cv2.BORDER_CONSTANT)]
    geometric_augmentation.append(A.OneOf(distortions, p=0.25))
​
    # rotate
    rotate = A.ShiftScaleRotate(p=0.25, scale_limit=0., rotate_limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT,
                                mask_value=0)
    geometric_augmentation.append(rotate)
​
    geometric_augmentation = A.Compose(geometric_augmentation)
    augmented = geometric_augmentation(image=image, mask=mask)
    image = augmented["image"]
    mask = augmented["mask"]
​
    # iaa affine and piecewise affine
    iaa_piecewise_affine = iaa.PiecewiseAffine(scale=(0.02, 0.04), nb_rows=(2, 3), nb_cols=(2, 3), order=1)
    iaa_affine = iaa.Affine(shear=(0, 5))
    iaa_aug = iaa.SomeOf((0, 1), [iaa_piecewise_affine, iaa_affine])
    image, mask = augment_image_and_mask(iaa_aug, image, mask)
​
    # scale and AR
    interpolations = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, ]
    interpolation = random.choice(interpolations)
    scale = A.RandomScale(scale_limit=(-0.25, 0.25), p=0.5, interpolation=interpolation)
    h, w, _ = image.shape
    h_percentage, w_percentage = (
        np.random.uniform(0.75, 1.25),
        np.random.uniform(0.75, 1.25),
    )
    interpolation = random.choice(interpolations)
    aspect_ratio = A.Resize(height=int(h_percentage * h),
                            width=int(w_percentage * w), p=0.5,
                            interpolation=interpolation)
​
    aug = A.OneOf([scale, aspect_ratio])
    augmented = aug(image=image, mask=mask)
    image = augmented["image"]
    mask = augmented["mask"]
    return image, mask
​
​
def non_geometric_pipeline(
    image: np.array,
    ) -> np.array:
    """Applies a pipeline of non geometric transformation to an image.
    Transformations are pixel_wise, and thus can be applied to image only.
​
​    Args:
        image : np.array, input image
​
    Returns:
        image: np.array, augmented image
    """
     
    # COLOR
    color = []
    color.append(A.RGBShift(p=0.5, r_shift_limit=30, g_shift_limit=30, b_shift_limit=30))
    color.append(A.RandomGamma(p=0.5, gamma_limit=(80, 120)))
    color.append(A.ToGray(p=0.5))
    color.append(A.CLAHE(p=0.5))
    color = A.OneOf(color, p=0.5)
​      
    # EXPOSURE
    exposure = []
    exposure.append(A.RandomBrightnessContrast())
    exposure.append(A.RandomShadow())
    exposure.append(A.RandomSunFlare())
    exposure.append(A.RandomToneCurve())
    color_alt = A.OneOf(exposure, p=0.5)


    # BLUR (motion, focal...)
    blur = []
    blur.append(A.GaussianBlur(p=1., blur_limit=3))
    blur.append(A.MotionBlur(p=1., blur_limit=3))
    blur = A.OneOf(p=0.25, transforms=blur)
​
    # NOISE
    noise = []
    noise.append(A.GaussNoise())
    noise.append(A.ISONoise(p=0.1, intensity=[0.1, 1.0]))
    noise = A.OneOf(noise, p=1)
​
    # compose blur + color + exposure + noise and apply to image
    aug = A.Compose([blur, color, exposure, noise], p=0.7)
    augmented = aug(image=image)
    image = augmented["image"]

    return image
​
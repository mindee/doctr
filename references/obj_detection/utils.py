# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


from random import randint
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import references.obj_detection.constants as const
import references.obj_detection.noise as noise
import references.obj_detection.shadow_ellipse as ellipse
import references.obj_detection.shadow_polygon as polygon
import references.obj_detection.shadow_single as single
import torch
from matplotlib.cm import get_cmap


class Add_noise():
    def __call__(self, img: torch.Tensor):
        img = img.permute(1, 2, 0).numpy() * 255
        if np.random.rand() > 0.35:
            # Adds random n shadows
            img = add_n_random_shadows(img)
        else:
            # Adds speckle noise of random intensity
            img = get_random_speckle_noise(img)
        return torch.from_numpy(img).permute(2, 1, 0) / 255


def add_n_random_shadows(image, n_shadow=4, blur_scale=1.0):
    intensity = np.random.uniform(const.MIN_SHADOW, const.MAX_SHADOW)
    return add_n_shadows(image, n_shadow, intensity, blur_scale)


def add_n_shadows(image, n_shadow=4, intensity=0.4, blur_scale=1.0):
    for i in range(n_shadow):
        blur_width = noise.get_blur_given_intensity(intensity, blur_scale)

        choice = np.random.uniform(0, 6)
        if choice < 1:
            image = polygon.add_n_triangles_shadow(image, intensity, blur_width)
        elif choice < 2:
            image = polygon.add_n_triangles_light(image, intensity, blur_width)
        elif choice < 3:
            image = single.add_single_light(image, intensity, blur_width)
        elif choice < 4:
            image = single.add_single_shadow(image, intensity, blur_width)
        elif choice < 5:
            image = ellipse.add_ellipse_light(image, intensity, blur_width)
        else:
            image = ellipse.add_ellipse_shadow(image, intensity, blur_width)

    return image


def add_n_random_blur(image, n=randint(1, 4)):
    for i in range(n):
        choice = np.random.uniform(0, 4)
        if choice < 1:
            image = blur(image, randint(1, 3))
        elif choice < 2:
            image = get_gauss_noise(image, randint(1, 100))
        elif choice < 3:
            image = get_saltpepper_noise(image, np.random.uniform(0.0001, 0.001))
        elif choice < 4:
            image = get_speckle_noise(image, np.random.uniform(0.01, 0.3))
    return image


def add_random_blur(image):
    max_blur = const.MAX_BLUR
    max_dimension = max(image.shape)
    if max_dimension < 2000 and max_blur > 2:
        max_blur -= 1
        if max_dimension < 1000 and max_blur > 1:
            max_blur -= 1
    intensity = randint(const.MIN_BLUR, max_blur)
    return blur(image, width=intensity)


def blur(image, width=9):
    for i in range(0, width):
        size = 2 ** i + 1
        image = cv2.blur(image, (size, size))
    return image


def get_blur_given_intensity(intensity, blur_scale):
    intensity = intensity * blur_scale
    if intensity < 0.4:
        return 5
    elif intensity < 0.5:
        return 6
    return 7


def add_random_gauss_noise(image):
    intensity = randint(const.MIN_GAUSS_NOISE, const.MAX_GAUSS_NOISE)
    return get_gauss_noise(image, intensity)


def get_gauss_noise(image, intensity=1):
    mean = 0
    sigma = intensity ** 0.5

    if len(image.shape) > 2:
        h, w, ch = image.shape
        gauss = np.random.normal(mean, sigma, (h, w, ch))
        gauss = gauss.reshape(h, w, ch)

    else:
        h, w = image.shape
        gauss = np.random.normal(mean, sigma, (h, w))
        gauss = gauss.reshape(h, w)

    gauss = image + gauss
    gauss = __get_normalized_image(gauss)
    return gauss


def __get_normalized_image(image):
    min_matrix = 0 * image
    max_matrix = min_matrix + 255
    image = np.minimum(image, max_matrix)
    image = np.maximum(image, min_matrix)
    return image.astype(np.uint8)


def add_random_saltpepper_noise(image):
    intensity = np.random.uniform(const.MIN_SALT_PEPPER_NOISE,
                                  const.MAX_SALTPEPPER_NOISE)
    return get_saltpepper_noise(image, intensity)


# tip: use it as first transformation, apply other noises afterwards
def get_saltpepper_noise(image, intensity=0.001, add_blur=const.ADD_BLUR_AFTER_SP_NOISE):
    s_vs_p = 0.5
    saltpepper = np.copy(image)
    num_salt = np.ceil(intensity * image.size * s_vs_p)
    coords = __get_coordinates_saltpepper(image, num_salt)
    saltpepper[coords] = 255
    num_pepper = np.ceil(intensity * image.size * (1. - s_vs_p))
    coords = __get_coordinates_saltpepper(image, num_pepper)
    saltpepper[coords] = 0

    if add_blur:
        return blur(saltpepper, width=1)
    return saltpepper


def __get_coordinates_saltpepper(image, num_salt):
    return tuple([np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape[: 2]])


def get_random_speckle_noise(image):
    intensity = np.random.uniform(const.MIN_SPECKLE_NOISE, const.MAX_SPECKLE_NOISE)
    return get_speckle_noise(image, intensity)


# tip: use it as first transformation, apply other noises afterwards
def get_speckle_noise(image, intensity=0.1, add_blur=const.ADD_BLUR_AFTER_SPECKLE_NOISE):
    intensity *= 127.5
    if len(image.shape) > 2:
        h, w, ch = image.shape
        speckle = -intensity / 2 + np.random.randn(h, w, ch) * intensity
        speckle = speckle.reshape(h, w, ch)
    else:
        h, w = image.shape
        speckle = -intensity / 2 + np.random.randn(h, w) * intensity
        speckle = speckle.reshape(h, w)

    speckle = image + speckle
    speckle = __get_normalized_image(speckle)
    if add_blur and intensity > 26:
        return blur(speckle, width=1)
    return speckle


def plot_samples(images, targets: List[Dict[str, np.ndarray]], classes: List[str]) -> None:
    cmap = get_cmap('gist_rainbow', len(classes))
    # Unnormalize image
    nb_samples = min(len(images), 4)
    _, axes = plt.subplots(1, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)
        target = img.copy()
        for box, class_idx in zip(targets[idx]['boxes'].numpy(), targets[idx]['labels']):
            r, g, b, _ = cmap(class_idx.numpy())
            color = int(round(255 * r)), int(round(255 * g)), int(round(255 * b))
            cv2.rectangle(target, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            text_size, _ = cv2.getTextSize(classes[class_idx], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_w, text_h = text_size
            cv2.rectangle(target, (int(box[0]), int(box[1])), (int(box[0]) + text_w, int(box[1]) - text_h), color, -1)
            cv2.putText(target, classes[class_idx], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)

        axes[idx].imshow(target)
    # Disable axis
    for ax in axes.ravel():
        ax.axis('off')
    plt.show()


def plot_recorder(lr_recorder, loss_recorder, beta: float = 0.95, **kwargs) -> None:
    """Display the results of the LR grid search.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py

    Args:
        lr_recorder: list of LR values
        loss_recorder: list of loss values
        beta (float, optional): smoothing factor
    """

    if len(lr_recorder) != len(loss_recorder) or len(lr_recorder) == 0:
        raise AssertionError("Both `lr_recorder` and `loss_recorder` should have the same length")

    # Exp moving average of loss
    smoothed_losses = []
    avg_loss = 0.
    for idx, loss in enumerate(loss_recorder):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

    # Properly rescale Y-axis
    data_slice = slice(
        min(len(loss_recorder) // 10, 10),
        -min(len(loss_recorder) // 20, 5) if len(loss_recorder) >= 20 else len(loss_recorder)
    )
    vals = np.array(smoothed_losses[data_slice])
    min_idx = vals.argmin()
    max_val = vals.max() if min_idx is None else vals[:min_idx + 1].max()  # type: ignore[misc]
    delta = max_val - vals[min_idx]

    plt.plot(lr_recorder[data_slice], smoothed_losses[data_slice])
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Training loss')
    plt.ylim(vals[min_idx] - 0.1 * delta, max_val + 0.2 * delta)
    plt.grid(True, linestyle='--', axis='x')
    plt.show(**kwargs)

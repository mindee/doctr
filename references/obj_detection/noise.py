#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from random import randint

import cv2
import numpy as np

import constants as const


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
def get_saltpepper_noise(image, intensity=0.0001, add_blur=const.ADD_BLUR_AFTER_SP_NOISE):
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

    # -intensity/2 <= speckle <= intensity/2
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

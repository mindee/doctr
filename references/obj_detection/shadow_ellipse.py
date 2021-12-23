import constants as const
import cv2
import numpy as np
import shadow_mask as mask


def add_n_ellipses_light(image, intensity=0.5, blur_width=6, n=1):
    inverted_colors = const.WHITE - image
    inverted_shadow = add_n_ellipses_shadow(inverted_colors, intensity, blur_width, n)

    return const.WHITE - inverted_shadow


def add_n_ellipses_shadow(image, intensity=0.5, blur_width=6, n=1):
    for i in range(n):
        image = add_ellipse_shadow(image,
                                   intensity=intensity,
                                   blur_width=blur_width,
                                   )

    return image


def add_ellipse_light(image, intensity=0.5, blur_width=6):
    inverted_colors = const.WHITE - image
    inverted_shadow = add_ellipse_shadow(inverted_colors, intensity, blur_width)

    return const.WHITE - inverted_shadow


def add_ellipse_shadow(image, intensity=0.5, blur_width=6):
    shadow_mask = np.zeros(image.shape[: 2], dtype=np.uint8)
    shadow_mask.fill(const.WHITE)
    ellipse = __get_multiple_ellipses(shadow_mask)

    return mask.apply_shadow_mask(image, blur_width, intensity, ellipse)


def __get_multiple_ellipses(image):
    h, w = image.shape[: 2]
    center = int(w * np.random.uniform()), int(h * np.random.uniform())
    random_h = np.random.uniform() * h
    random_w = np.random.uniform() * w
    axes1 = int(random_h * 0.2), int(random_w * 0.2)
    axes2 = int(random_h * 0.4), int(random_w * 0.4)
    axes3 = int(random_h * 0.6), int(random_w * 0.6)
    axes4 = int(random_h * 0.8), int(random_w * 0.8)
    axes5 = int(random_h), int(random_w)
    angle = 360 * np.random.uniform()

    ellipse = get_single_ellipse(image, center, axes5, angle, const.DARK_WHITE)
    ellipse = get_single_ellipse(ellipse, center, axes4, angle, const.LIGHT_GRAY)
    ellipse = get_single_ellipse(ellipse, center, axes3, angle, const.GRAY)
    ellipse = get_single_ellipse(ellipse, center, axes2, angle, const.DARK_GRAY)

    return get_single_ellipse(ellipse, center, axes1, angle, const.LIGHT_BLACK)


def get_single_ellipse(image, center, axes, angle, color):
    start_angle = 0
    end_angle = 360
    thickness = -1

    return cv2.ellipse(image, center, axes, angle, start_angle, end_angle,
                       color, thickness)

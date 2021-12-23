import channels as channels
import cv2
import noise as noise
import numpy as np


def apply_shadow_mask(image, blur_width, intensity, shadow_mask):
    shadow_mask = __normalize_shadow_mask(blur_width, intensity, shadow_mask)
    if len(image.shape) > 2:
        blue, green, red = channels.get_bgr_channels(image)
        blue = __apply_mask_to_channel(blue, shadow_mask)
        green = __apply_mask_to_channel(green, shadow_mask)
        red = __apply_mask_to_channel(red, shadow_mask)
        return cv2.merge((blue, green, red))

    return __apply_mask_to_channel(image, shadow_mask)


def __apply_mask_to_channel(image, shadow_mask):
    image = np.multiply(image, shadow_mask)
    return image.astype(np.uint8)


def __normalize_shadow_mask(blur_width, intensity, shadow_mask):
    shadow_mask = noise.blur(shadow_mask, blur_width)
    normalized_mask = (shadow_mask / 250.0)
    normalized_mask = normalized_mask + (1.0 - normalized_mask) * (1.0 - intensity)
    ones_matrix = 0 * shadow_mask + 1.0
    normalized_mask = np.minimum(normalized_mask, ones_matrix)
    return normalized_mask

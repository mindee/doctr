#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from random import randint

import constants as const
import numpy as np
import shadow_mask as mask


def add_single_light(image, intensity = 0.5, blur_width = 8):
	inverted_colors = const.WHITE - image
	inverted_shadow = add_single_shadow(inverted_colors, intensity, blur_width)

	return const.WHITE - inverted_shadow


def add_single_shadow(image, intensity = 0.5, blur_width = 8):
	h, w = image.shape[ : 2]
	top_y = __get_random_number(w)
	top_x = __get_random_number(h)
	bot_x = __get_random_number(h)
	bot_y = __get_random_number(w)
	x_m = np.mgrid[0 : h, 0 : w][0]
	y_m = np.mgrid[0 : h, 0 : w][1]
	if len(image.shape) > 2:
		shadow_mask = 0 * image[ :, :, 0]
	else:
		shadow_mask = 0 * image

	mask_dark_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
	shadow_mask[mask_dark_gray >= 0] = const.DARK_GRAY

	space = 50
	if bot_x < top_x and top_y > bot_y:
		shadow_mask[mask_dark_gray < 0] = const.LIGHT_BLACK
		top_x -= __get_random_space(space)
		bot_x -= __get_random_space(space)
		top_y += __get_random_space(space)
		bot_y += __get_random_space(space)
		mask_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_gray >= 0] = const.GRAY
		top_x -= __get_random_space(space)
		bot_x -= __get_random_space(space)
		top_y += __get_random_space(space)
		bot_y += __get_random_space(space)
		mask_light_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_light_gray >= 0] = const.LIGHT_GRAY
		top_x -= __get_random_space(space)
		bot_x -= __get_random_space(space)
		top_y += __get_random_space(space)
		bot_y += __get_random_space(space)
		mask_dark_white = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_dark_white >= 0] = const.DARK_WHITE

	if bot_x < top_x and top_y < bot_y:
		shadow_mask[mask_dark_gray < 0] = const.LIGHT_BLACK
		top_x += __get_random_space(space)
		bot_x += __get_random_space(space)
		top_y += __get_random_space(space)
		bot_y += __get_random_space(space)
		mask_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_gray >= 0] = const.GRAY
		top_x += __get_random_space(space)
		bot_x += __get_random_space(space)
		top_y += __get_random_space(space)
		bot_y += __get_random_space(space)
		mask_light_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_light_gray >= 0] = const.LIGHT_GRAY
		top_x += __get_random_space(space)
		bot_x += __get_random_space(space)
		top_y += __get_random_space(space)
		bot_y += __get_random_space(space)
		mask_dark_white = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_dark_white >= 0] = const.DARK_WHITE

	if bot_x > top_x and top_y > bot_y:
		shadow_mask[mask_dark_gray < 0] = const.LIGHT_BLACK
		top_x -= __get_random_space(space)
		bot_x -= __get_random_space(space)
		top_y -= __get_random_space(space)
		bot_y -= __get_random_space(space)
		mask_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_gray >= 0] = const.GRAY
		top_x -= __get_random_space(space)
		bot_x -= __get_random_space(space)
		top_y -= __get_random_space(space)
		bot_y -= __get_random_space(space)
		mask_light_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_light_gray >= 0] = const.LIGHT_GRAY
		top_x -= __get_random_space(space)
		bot_x -= __get_random_space(space)
		top_y -= __get_random_space(space)
		bot_y -= __get_random_space(space)
		mask_dark_white = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_dark_white >= 0] = const.DARK_WHITE

	if bot_x > top_x and top_y < bot_y:
		shadow_mask[mask_dark_gray < 0] = const.LIGHT_BLACK
		top_x += __get_random_space(space)
		bot_x += __get_random_space(space)
		top_y -= __get_random_space(space)
		bot_y -= __get_random_space(space)
		mask_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_gray >= 0] = const.GRAY
		top_x += __get_random_space(space)
		bot_x += __get_random_space(space)
		top_y -= __get_random_space(space)
		bot_y -= __get_random_space(space)
		mask_light_gray = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_light_gray >= 0] = const.LIGHT_GRAY
		top_x += __get_random_space(space)
		bot_x += __get_random_space(space)
		top_y -= __get_random_space(space)
		bot_y -= __get_random_space(space)
		mask_dark_white = __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m)
		shadow_mask[mask_dark_white >= 0] = const.DARK_WHITE

	return mask.apply_shadow_mask(image, blur_width, intensity, shadow_mask)


def __get_mask_condition(bot_x, bot_y, top_x, top_y, x_m, y_m):
	return (x_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (y_m - top_y)


def __get_random_number(number):
	return number * np.random.uniform()


def __get_random_space(space):
	return space + space * np.random.uniform()

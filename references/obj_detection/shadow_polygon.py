#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from random import randint

import constants as const
import cv2
import numpy as np
import shadow_mask as mask


def add_n_triangles_light(image, intensity = 0.5, blur_width = 6, n = 1):
	inverted_colors = const.WHITE - image
	inverted_shadow = add_n_triangles_shadow(inverted_colors, intensity, blur_width, n)
	return const.WHITE - inverted_shadow


def add_n_triangles_shadow(image, intensity = 0.5, blur_width = 6, n = 1):
	for i in range(n):
		image = add_polygon_shadow(image,
					   n_sides = 3,
					   intensity = intensity,
					   blur_width = blur_width,
					   )

	return image


# tip: just stick with triangles, other polygons have incoherent shades
def add_polygon_light(image, n_sides = 3, intensity = 0.5, blur_width = 6):
	inverted_colors = const.WHITE - image
	inverted_shadow = add_polygon_shadow(inverted_colors, n_sides, intensity, blur_width)
	return const.WHITE - inverted_shadow

# tip: just stick with triangles, other polygons have incoherent shades
def add_polygon_shadow(image, n_sides = 3, intensity = 0.5, blur_width = 6):
	if len(image.shape) > 2:
		shadow_mask = 0 * image[:, :, 0] + const.WHITE
	else:
		shadow_mask = 0 * image + const.WHITE

	points = __get_points(n_sides, shadow_mask)

	centre = __get_centre(points)
	points1 = __scale_points(points.copy(), centre, 0.5)
	points2 = __scale_points(points.copy(), centre, 1.5)
	points3 = __scale_points(points.copy(), centre, 2)
	points4 = __scale_points(points.copy(), centre, 2.5)

	'''
	This line shouldn't be necessary, but it solves the error
	"typeerror: the layout array is incompatible with..."
	in some rare images.
	This is due to a bug in the python opencv wrapper.
	It may be needed also elsewhere.
	'''
	shadow_mask = shadow_mask.copy()
	cv2.fillPoly(shadow_mask, [points4], const.DARK_WHITE)
	cv2.fillPoly(shadow_mask, [points3], const.LIGHT_GRAY)
	cv2.fillPoly(shadow_mask, [points2], const.GRAY)
	cv2.fillPoly(shadow_mask, [points] , const.DARK_GRAY)
	cv2.fillPoly(shadow_mask, [points1], const.LIGHT_BLACK)

	return mask.apply_shadow_mask(image, blur_width, intensity, shadow_mask)


def __get_points(n_sides, image):
	points = []
	h, w = image.shape[:2]
	for i in range(n_sides):
		points.append([np.int32(w * 1.5 * np.random.uniform() - w * 0.25),
			       np.int32(h * 1.5 * np.random.uniform() - h * 0.25)],
			      )

	points = np.array(points)
	points = points.reshape((-1, 1, 2))
	return points


def __scale_points(points, centre, scale):
	for p in points:
		__scale_point(p[0], centre, scale = scale + np.random.uniform()/2)

	return points


def __scale_point(point, centre, scale = 0.1):
	point[0] = int(round(centre[0] + (point[0] - centre[0]) * scale))
	point[1] = int(round(centre[1] + (point[1] - centre[1]) * scale))


def __get_centre(points):
	sum_points = sum(points[1:], points[0])[0]
	n_points = len(points)
	return (
		__get_average(n_points, sum_points[0]),
		__get_average(n_points, sum_points[1]),
		)


def __get_average(n_points, sum):
	return int(round(sum / n_points))

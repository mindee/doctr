import random

from enum import Enum

N_FILES_OUTPUT = 100

# allowed input images extensions
IMAGE_EXTENSIONS = [".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".jp2",
                    ".dib",
                    ".webp",
                    ".sr",
                    ".ras",
                    ".tiff",
                    ".tif",
                    ".pbm",
                    ".pgm",
                    ".ppm",
                    ]

# default input files path
FILES_INPUT_PATH = "./input/"

# default output files path
FILES_OUTPUT_PATH = "./output/"

'''
max number of transformations
that are randomly applied for each output image.
If N_TRANSFORMATIONS > total number of transformations
specified in the MaxTransformation field
N_TRANSFORMATIONS will be reset as sum_MaxTransformation_fields
'''
N_TRANSFORMATIONS = 3

'''
MaxTransformation contains, for each transformation,
the maximum number of times that each transformation is performed.
Useful to apply some transformations more often ( n > 1 )
or to exclude them` altogether ( n = 0 )
'''


class MaxTransformation:
    SALT_PEPPER_NOISE = 1
    SPECKLE_NOISE = 1
    GAUSS_NOISE = 1
    BLUR = 1

    SHADOW = 1
    ENHANCEMENTS = 1
    SHADE_COLOR = 1

    # The following transformations
    # will alter pixel coordinates
    SHEAR = 1
    SKEW = 1
    WARP = 1
    ROTATION = 1


# MIN/MAX AVG BLURRING
MIN_BLUR = 1
MAX_BLUR = 3

# MIN/MAX GAUSS NOISE
MIN_GAUSS_NOISE = 1
MAX_GAUSS_NOISE = 100

# MIN/MAX SALT AND PEPPER NOISE
MIN_SALT_PEPPER_NOISE = 0.0001
MAX_SALTPEPPER_NOISE = 0.001

# MIN/MAX SPECKLE
MIN_SPECKLE_NOISE = 0.01
MAX_SPECKLE_NOISE = 0.3

# MIN/MAX SHADOW
MIN_SHADOW = 0.3
MAX_SHADOW = 0.7

# MIN/MAX IMAGE BRIGHTNESS
MIN_BRIGHTNESS = 0.6
MAX_BRIGHTNESS = 1.4

# MIN/MAX IMAGE CONTRAST
MIN_CONTRAST = 0.5
MAX_CONTRAST = 1.7

# MIN/MAX IMAGE SHARPNESS
MIN_SHARPNESS = 0.1
MAX_SHARPNESS = 5.0

# MIN/MAX COLOR SHADING
MIN_COLOR_SHADE = 0.06
MAX_COLOR_SHADE = 0.35

# MAX SHEAR DISTORTION
MAX_SHEAR = 0.05

# MAX SKEW DISTORTION
MAX_SKEW = 0.05

# MIN/MAX WARP DISTORTION
MIN_WARP = 14
MAX_WARP = 51

# MIN/MAX ROTATION ANGLE
MAX_ANGLE = 0.02

# By default salt&pepper and speckle noise
# is followed by blurring
ADD_BLUR_AFTER_SPECKLE_NOISE = True
ADD_BLUR_AFTER_SP_NOISE = True

READ_IMAGE_AS_GRAYSCALE = False

BLACK = 0
LIGHT_BLACK = 50
DARK_GRAY = 100
GRAY = 150
LIGHT_GRAY = 200
DARK_WHITE = 250
WHITE = 255

SEED = random.random()
# Max integer value for Python 2. Integers in Python 3 are unbounded
MAX_RANDOM = 2147483648


class Enhancement(Enum):
    brightness = 0
    contrast = 1
    sharpness = 2

    @staticmethod
    def get_random():
        return random.choice(list(Enhancement))


class Channels(Enum):
    bgr = 0
    hsv = 1
    hls = 2

    @staticmethod
    def get_random():
        return random.choice(list(Channels))

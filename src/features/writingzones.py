import logging

import numpy as np

from src.features.models import WritingZoneFeatures
from src.models import TextImage


def get_writing_zone_features(text_image: TextImage):
    logging.info("Extracting writing zone features")
    f1, f2, f3 = get_writing_zones(text_image.image)
    logging.info("Writing zone features successfully extracted")

    return WritingZoneFeatures(f1, f2, f3)


def get_writing_zones(image):
    black_white = list(map(lambda x: get_black_pixels_in_row_count(x, 8, 150), image))
    ub, lb, err = get_minimum_error_bounds(black_white)
    height = len(image)

    f1 = (height - ub) / height  # upper writing zone as a fraction of the total writing zone
    f2 = (ub - lb) / height  # middle zone as a fraction of the total writing zone
    f3 = lb / height  # lower zone as a fraction of the total writing zone

    return f1, f2, f3


def get_black_pixels_in_row_count(row, bits, slack):
    count = 0
    for pixel in row:
        if pixel < (2 ** bits + slack) / 2:
            count += 1
    return count


def get_minimum_error_bounds(pixels_per_row):
    height = len(pixels_per_row)
    minimum = (-1, -1, np.inf)
    for ub in range(1, height):
        for lb in range(1, ub):
            err = hist_error(ub, lb, pixels_per_row)
            if err < minimum[2]:
                minimum = (ub, lb, err)
    return minimum


def hist_error(ub, lb, pixels_per_row):
    """
    :param ub:              Upper baseline. Integer value.
    :param lb:              Lower baseline. Integer value.
    :param pixels_per_row:  An array containing the amount of black pixels for each row in the image
    :return:                The error between an ideal hist and pixels_per_row
    """
    height = len(pixels_per_row)
    total = sum(pixels_per_row)
    lower_ideal, middle_ideal, upper_ideal = ideal_hist(height, ub, lb, total, 0.8)
    lower_img, middle_img, upper_img = pixels_per_row[:lb], pixels_per_row[lb:ub], pixels_per_row[ub:]
    error1 = sum(map(lambda x: (x - lower_ideal) ** 2, lower_img))
    error2 = sum(map(lambda x: (x - middle_ideal) ** 2, middle_img))
    error3 = sum(map(lambda x: (x - upper_ideal) ** 2, upper_img))
    return error1 + error2 + error3


def ideal_hist(height, ub, lb, total, threshold):
    """
    For an 'ideal histogram', the following is assumed:
    - threshold% of the total amount of pixels is between ub and lb.
    - (100-threshold)/2% is between top line (height) and ub
    - (100-threshold)/2% is between bottom line (0) and lb
    :param height:      Height of the image. Integer value
    :param ub:          Upper baseline. Integer value.
    :param lb:          Lower baseline. Integer value.
    :param total:       Total amount of black pixels in the image. Integer value.
    :param threshold:   Percentage used to calculate amount of pixels in each zone. float between 0 and 1.
    :return:            an ideal histogram as a triple, representing the average amount of pixels for the lower zone,
                        middle zone, and upper zone, respectively
    """
    middle_zone = int(round(threshold * total))
    upper_lower_zone = int(round(((1 - threshold) / 2) * total))

    # number of pixels per row in each zone
    middle = int(round(middle_zone / (ub - lb)))
    upper = int(round(upper_lower_zone / (height - ub)))
    lower = int(round(upper_lower_zone / lb))

    return lower, middle, upper

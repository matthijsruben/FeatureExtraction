import logging
import operator

import numpy as np
import statsmodels.api as sm

from src.preparation import utils
from src.preparation.features.models import ContourFeatures
from src.preparation.models import TextImage


def get_contour_features(text_image, extremes_window=10, slope_distance=10):
    """
    Generates and returns the features related to the upper and lower characteristic contour of the handwritten text in
    the image.
    :param text_image: the image to generate the features for.
    :param extremes_window: the size of the neighborhood in which local extremes have to be the maximum (minimum)
    :param slope_distance: the distance to the right/left of the extreme to find the slope for
    :return: A tuple consisting of two ContourFeatures objects. First the lower contour features, then the upper
    contour features.
    """
    logging.info("Extracting contour features")
    lc_features = _get_contour_features(text_image, True, extremes_window, slope_distance)
    uc_features = _get_contour_features(text_image, False, extremes_window, slope_distance)
    logging.info("Contour features successfully extracted")

    return lc_features, uc_features


def _get_contour_features(text_image: TextImage, lower, extremes_window, slope_distance):
    """

    :param text_image: an Image to generate the features for
    :param lower: whether to generate lower contour features (True) or upper contour features (false)
    :param extremes_window: the size of the neighborhood in which local extremes have to be the maximum (minimum)
    :param slope_distance: the distance to the right/left of the extreme to find the slope for
    :return: A ContourFeatures object.
    """
    contour = get_characteristic_contour(text_image.bw_image, lower)
    _, slant, slant_mse = get_characteristic_contour_polynomial(contour)
    local_maxima = get_local_extremes(contour, extremes_window, True)
    local_maxima_freq = get_extremes_frequency(contour, local_maxima)
    local_minima = get_local_extremes(contour, extremes_window, False)
    local_minima_freq = get_extremes_frequency(contour, local_minima)

    maxima_slopes_left, maxima_slopes_right = get_local_slopes(contour, local_maxima, slope_distance)
    maxima_slopes_left_avg = np.average(maxima_slopes_left)
    maxima_slopes_right_avg = np.average(maxima_slopes_right)
    minima_slopes_left, minima_slopes_right = get_local_slopes(contour, local_minima, slope_distance)
    minima_slopes_left_avg = np.average(minima_slopes_left)
    minima_slopes_right_avg = np.average(minima_slopes_right)

    return ContourFeatures(
        slant=slant,
        slant_mse=slant_mse,
        local_max_freq=local_maxima_freq,
        local_min_freq=local_minima_freq,
        max_slopes_left_avg=maxima_slopes_left_avg,
        max_slopes_right_avg=maxima_slopes_right_avg,
        min_slopes_left_avg=minima_slopes_left_avg,
        min_slopes_right_avg=minima_slopes_right_avg,
        lower=lower
    )


def get_characteristic_contour(bw_image, lower):
    """
    Returns the characteristic contour of the provided image after elimination of discontinuities in y-direction, in a
    normalized format such that the returned values are always >= 0.
    :param bw_image: a black-and-white (binary) image to find the contours in.
    :param lower: whether to extract the lower contour (True) or the upper contour (False).
    :return: The normalized contour values.
    """
    contours = []
    last_contour_y = None

    for x in range(0, len(bw_image[0])):
        column = utils.extract_column(bw_image, x)
        contour_y = utils.find_lower_boundary_pixel(column) if lower else utils.find_upper_boundary_pixel(column)

        if contour_y is None:
            # Eliminate gaps between words or parts of a word
            continue
        elif last_contour_y is None:
            contours.append(contour_y)
        else:
            contours.append(contours[-1] + np.sign(contour_y - last_contour_y))

        last_contour_y = contour_y

    min_y = min(0, min(contours))
    return [y - min_y for y in contours]


def get_characteristic_contour_polynomial(contour):
    """
    Performs a regression analysis to find the parameters of the first-degree polynomial that best fits the
    characteristic contour.
    :param contour: an array of the y-coordinates of the characteristic contour.
    :return: Considering formula y = ax + b, returns a tuple of (b; a; mean-squared-error).
    """
    # Create an array with x-coordinate for each data point
    xs = np.arange(0, len(contour), 1)
    # Add an intercept (constant)
    xs = sm.add_constant(xs)

    model = sm.OLS(contour, xs)
    results = model.fit()

    return results.params[0], results.params[1], results.mse_resid


def get_local_extremes(contour, window, maxima=True):
    """
    Finds the x-coordinates of the local extremes, defined as points on the characteristic contour such that there is no
    other point within a neighbourhood of the given size that has a larger (smaller) y-value.
    :param contour: the contours to find extremes in.
    :param window: the size of the neighborhood in which the point has to be the maximum (minimum)
    :param maxima: whether to find the maxima (True) or the minima (False)
    :return: The array of x-coordinates of the found local extremes.
    """
    extreme = max if maxima else min
    comparator = operator.ge if maxima else operator.le

    extremes = []
    prev_was_extreme = False

    for i in range(0, len(contour)):
        # Find the extreme value in the neighborhood to the left of the current coordinate
        extreme_left = None
        if i > 0:
            extreme_left = extreme(contour[max(0, i - window):max(0, i)])

        # Find the extreme value to the right of the current coordinate
        extreme_right = None
        if i < len(contour) - 1:
            extreme_right = extreme(contour[min(i + 1, len(contour)):min(i + 1 + window, len(contour))])

        if (extreme_left is None or comparator(contour[i], extreme_left)) and (
                extreme_right is None or comparator(contour[i], extreme_right)):
            # Ensure that an extreme that covers multiple x-coordinates is only added once
            if not prev_was_extreme:
                extremes.append(i)
            prev_was_extreme = True
        else:
            prev_was_extreme = False

    return extremes


def get_local_slopes(contour, extremes, distance):
    """
    Performs a regression analysis to find the slopes of the first-degree polynomials that best fits the characteristic
    contour to the left (right) of the given extremes.
    :param contour: the contour to find the slopes for
    :param extremes: the x-coordinates of the extremes around which to find the slopes
    :param distance: the distance to the right/left of the extreme to find the slope for
    :return: A tuple consisting of (slopes to the left of extremes, slopes to the right of extremes).
    """
    local_slopes_left = []
    local_slopes_right = []
    for extreme in extremes:
        if extreme > 0:
            contour_part_left = contour[max(0, extreme - distance):extreme]
            _, slant, _ = get_characteristic_contour_polynomial(contour_part_left)
            local_slopes_left.append(slant)

        if extreme < len(contour) - 1:
            contour_part_right = contour[extreme + 1:min(len(contour), extreme + 1 + distance)]
            _, slant, _ = get_characteristic_contour_polynomial(contour_part_right)
            local_slopes_right.append(slant)

    return local_slopes_left, local_slopes_right


def get_extremes_frequency(contour, extremes):
    """
    :param contour: the contour in which the extremes exist.
    :param extremes: the x-coordinates of extremes in the contour
    :return: The frequency of the extremes when compared to the contour.
    """
    return len(extremes) / len(contour)

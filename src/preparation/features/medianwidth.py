import logging
import statistics

from src.preparation import utils
from src.preparation.features.models import MedianWidthFeatures
from src.preparation.models import TextImage
from src.preparation.utils import get_bw_transition_count


def get_median_width_features(text_image: TextImage):
    logging.info("Extracting median-width features")
    median_width = get_median_width(text_image.image)
    logging.info("Median-width features successfully extracted")

    return MedianWidthFeatures(median_width)


def get_median_width(image):
    transitions = list(map(get_bw_transition_count, image))
    row_most_transitions = image[transitions.index(max(transitions))]
    prev = False
    dist = 0
    distances = []

    for pixel in row_most_transitions:
        white = utils.get_bw_pixel(pixel, 8, 150) == 255

        if white and (dist == 0 or (dist > 0 and prev)):
            dist += 1
        # Exclude whitespace after the last black pixel
        elif not white and prev:
            distances.append(dist)
            dist = 0

        prev = white

    # Exclude whitespace before the first black pixel
    if utils.get_bw_pixel(row_most_transitions[0], 8, 150) == 255:
        distances.pop(0)

    return statistics.median(distances)

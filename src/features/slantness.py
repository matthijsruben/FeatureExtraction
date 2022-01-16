import functools
import logging
import statistics

import math

from src import utils
from src.features.models import SlantnessFeatures
from src.models import TextImage


def get_slantness_features(text_image: TextImage):
    logging.info("Extracting slantness features")
    max_angle, avg_angle, stdev_angle = get_slantness(text_image.bw_image)
    logging.info("Slantness features successfully extracted")

    return SlantnessFeatures(max_angle, avg_angle, stdev_angle)


def get_slantness(image):
    amount_of_angles = 40
    length = len(image)
    score = {}

    # Loop through all angles
    for i in range(1, amount_of_angles):
        angle = i * (math.pi / amount_of_angles)
        line = find_line(angle, length)
        score[angle] = 0

        # For each angle, loop through all shifts (shift 10 pixels everytime)
        for shift in range(0, len(image[0])):
            shifted_line = list(map(lambda pix: (pix[0] + shift, pix[1]), line))
            filtered_line = list(
                filter(lambda pix: 0 <= pix[0] < len(image[0]) and 0 <= pix[1] < len(image), shifted_line))
            total = len(filtered_line)
            img_line = list(map(lambda pix: image[pix[1]][pix[0]], filtered_line))

            # Calculate # transitions and # black pixels on the line
            transitions = utils.get_bw_transition_count(img_line)
            blacks = len(list(filter(lambda pix: pix == 0, img_line)))

            # Only lines (> 3/4 * height of image); consisting of >=50% black pixels, with 1 consecutive piece of black
            if transitions <= 2 and blacks / total >= 0.5 and total >= 3 * len(image) / 4:
                score[angle] += 1

    sample = functools.reduce(lambda a, b: a + b, list(map(lambda x: [x[0]] * x[1], score.items())))
    stdev_angle = round(math.degrees(statistics.stdev(sample)), 1) if len(sample) > 1 else 0
    avg_angle = round(180 - math.degrees(statistics.mean(sample)), 1) if len(sample) >= 1 else 0
    max_angle = round(180 - math.degrees(max(score, key=score.get)), 1)

    return max_angle, avg_angle, stdev_angle


def find_line(angle, length):
    line = set()

    for i in range(0, length + 1):
        vertical = i * math.sin(angle)
        horizontal = i * math.cos(angle)
        pixel = (round(horizontal), round(vertical))
        line.add(pixel)

    return sorted(list(line))

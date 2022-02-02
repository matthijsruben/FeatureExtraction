import math
import numpy as np
from matplotlib import pyplot as plt

from src.preparation.features.slantness import find_line


def visualize_contour(contour):
    new_image = [[255 for pix in range(0, len(contour))] for row in range(0, max(contour) + 1)]
    for x, y in enumerate(contour):
        new_image[y][x] = 0

    plt.imshow(new_image)


def visualize_contour_extremes(contour, maxima, minima):
    maxima_y = [contour[x] for x in maxima]
    minima_y = [contour[x] for x in minima]

    marker_style = "x"
    marker_size = 20
    plt.scatter(maxima, maxima_y, s=marker_size, marker=marker_style)
    plt.scatter(minima, minima_y, s=marker_size, marker=marker_style)


def visualize_contour_slant(contour, intersection, slant):
    X = np.arange(0, len(contour), 1)
    Y = intersection + slant * X

    plt.plot(X, Y)


def visualize_image(pixels):
    plt.imshow(pixels, cmap='gray')
    plt.show()


def visualize_slantness(bw_image, slantness):
    # TODO: Move non-visualisation-related logic outside of this function
    image = bw_image.copy()
    line = find_line(math.radians(180 - slantness), len(image))

    for shift in range(0, len(image.bw_image[0]), 50):
        shifted_line = list(map(lambda pix: (pix[0] + shift, pix[1]), line))
        for pixel in shifted_line:
            if pixel[1] < len(image) and pixel[0] < len(image[0]):
                image[pixel[1]][pixel[0]] = 150

    visualize_image(image)


def visualize_baselines(bw_image, upper_baseline, lower_baseline):
    image = bw_image.copy()

    image[upper_baseline].fill(0)
    image[lower_baseline].fill(0)

    visualize_image(image)

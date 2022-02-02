import logging
import statistics

import numpy as np

from src.preparation import utils
from src.preparation.features.models import ComponentsBlobsFeatures
from src.preparation.models import TextImage


def get_components_and_blobs_features(text_image: TextImage):
    """
    Generates and returns the features related to the components and blobs in the handwritten text image.
    :param text_image:
    :return:
    """
    logging.info("Extracting components and blobs features")
    avg_distance, stdev_distance, avg_within_word_distance, avg_between_word_distance, \
    avg_area, avg_perimeter, avg_shape_factor, avg_roundness = features_components_blobs(text_image.image)
    logging.info("Components and blobs features successfully extracted")

    return ComponentsBlobsFeatures(avg_distance, stdev_distance, avg_within_word_distance, avg_between_word_distance,
                                   avg_area, avg_perimeter, avg_shape_factor, avg_roundness)


def has_white_neighbor(x, y, image, connectivity):
    if connectivity == 8:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= y + dy < len(image) and 0 <= x + dx < len(image[0]) and image[y + dy][x + dx] != 0 \
                        and not (dx == 0 and dy == 0):
                    return True
    elif connectivity == 4:
        neighbors = [(y + 1, x), (y, x + 1), (y - 1, x), (y, x - 1)]
        for nb in neighbors:
            if 0 <= nb[0] < len(image) and 0 <= nb[1] < len(image[0]) and image[nb[0]][nb[1]] != 0:
                return True
    else:
        print("Note that the connectivity must be either 4 (N, E, S, W) or 8 (N, NE, E, SE, S, SW, W, NW).")
    return False


def flood_fill(x, y, image, bounds=False):
    if not bounds:
        min_x, min_y, max_x, max_y = 0, 0, len(image[0]), len(image)
    else:
        min_x, min_y, max_x, max_y = bounds[0], bounds[1], bounds[2], bounds[3]

    queue = [(x, y)]
    visited = []
    while len(queue) != 0:
        cur = queue[0]
        queue.pop(0)
        if image[cur[1]][cur[0]] == image[y][x] and cur not in visited:
            if not (min_x <= cur[0] <= max_x and min_y <= cur[1] <= max_y):
                return visited, False
            visited.append(cur)
            # Add neighbors to the queue if they are pixels within the image
            neighbors = [(cur[0] - 1, cur[1]), (cur[0] + 1, cur[1]), (cur[0], cur[1] + 1), (cur[0], cur[1] - 1)]
            for nb in neighbors:
                if 0 <= nb[1] < len(image) and 0 <= nb[0] < len(image[0]):
                    queue.append(nb)
    return visited, True


def scan(lx, rx, y, queue, image, original_color, visited):
    added = False
    for x in range(lx, rx + 1):
        if image[y][x] != original_color or (x, y) in visited:
            added = False
        elif not added:
            queue.append((x, y))
            added = True


def scan_fill(x, y, image):
    original_color = image[y][x]
    queue = [(x, y)]
    visited = []
    while len(queue) != 0:
        x, y = queue[0][0], queue[0][1]
        queue.pop(0)
        lx = x
        while lx - 1 >= 0 and image[y][lx - 1] == original_color and (lx - 1, y) not in visited:
            visited.append((lx - 1, y))
            lx -= 1
        while x < len(image[0]) and image[y][x] == original_color and (x, y) not in visited:
            visited.append((x, y))
            x += 1
        if y + 1 < len(image):
            scan(lx, x - 1, y + 1, queue, image, original_color, visited)
        if y - 1 >= 0:
            scan(lx, x - 1, y - 1, queue, image, original_color, visited)
    return visited


def find_blobs_of_component(component, black_white, minimum_size, border_pixels):
    min_x, min_y = min(component)[0], min(component, key=lambda pix: pix[1])[1]
    max_x, max_y = max(component)[0], max(component, key=lambda pix: pix[1])[1]
    bounds = (min_x, min_y, max_x, max_y)
    visited = []
    blobs = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if black_white[y][x] != 0 and (x, y) not in visited:
                white_area, is_blob = flood_fill(x, y, black_white, bounds)
                visited.extend(white_area)
                # Check if the white area is a blob (is not background, is >= 20 pixels, is not touching border)
                if is_blob and len(white_area) >= minimum_size and not any(
                        [pixel in border_pixels for pixel in white_area]):
                    blobs.append(white_area)
    return blobs


def find_components_and_blobs(image):
    black_white = utils.get_bw_image(image, 200)
    border_pixels = utils.get_border_pixels(len(black_white[0]), len(black_white))
    visited = []
    all_blobs = []
    all_components = []
    for y in range(0, len(black_white)):
        for x in range(0, len(black_white[0])):
            if black_white[y][x] == 0 and (x, y) not in visited:
                # find connected black component using scan-fill algorithm
                component = scan_fill(x, y, black_white)
                # find the white blobs existing withing the bounds of that component
                blobs = find_blobs_of_component(component, black_white, 20, border_pixels)
                # append to overall information of the image
                all_components.append(component)
                all_blobs.extend(blobs)
                visited.extend(component)
    return all_components, all_blobs


def find_blob_perimeter(blob):
    min_x, min_y = min(blob)[0], min(blob, key=lambda pix: pix[1])[1]
    max_x, max_y = max(blob)[0], max(blob, key=lambda pix: pix[1])[1]
    blob = list(map(lambda pix: (pix[0] - min_x + 1, pix[1] - min_y + 1), blob))
    blob_image = [[255 for x in range(max_x - min_x + 3)] for y in range(max_y - min_y + 3)]
    for pixel in blob:
        x, y = pixel[0], pixel[1]
        blob_image[y][x] = 0
    circumference = list(filter(lambda pix: has_white_neighbor(pix[0], pix[1], blob_image, 4), blob))
    # IAMloader.show(blob_image)
    # Slower Method:
    # perimeter = len(list(filter(lambda pixel: len(list(filter(lambda nb: nb not in blob, [(pixel[0], pixel[1] + 1), (pixel[0] + 1, pixel[1]), (pixel[0], pixel[1] - 1), (pixel[0] - 1, pixel[1])]))) > 0, blob)))
    return len(circumference)


def features_components_blobs(image):
    # Calculate the components (connected black pixels) and the blobs (connected white pixels, enclosed by black pixels)
    components, blobs = find_components_and_blobs(image)

    # PART 1: COMPONENTS
    bounding_boxes = sorted(map(lambda comp: (min(comp)[0], min(comp, key=lambda pix: pix[1])[1],
                                              max(comp)[0], max(comp, key=lambda pix: pix[1])[1]), components))
    distances = list(map(lambda z: z[1][0] - z[0][2], zip(bounding_boxes, bounding_boxes[1:])))
    total_components = len(distances)
    if total_components > 0:
        avg_distance = sum(distances) / total_components
        stdev_distance = statistics.stdev(distances)
        threshold = avg_distance + stdev_distance / 3
        within_word_distances = list(filter(lambda dist: dist < threshold, distances))
        between_word_distances = list(filter(lambda dist: dist >= threshold, distances))
        total_within, total_between = len(within_word_distances), len(between_word_distances)
        if total_within > 0 and total_between > 0:
            avg_within_word_distance = sum(within_word_distances) / total_within
            avg_between_word_distance = sum(between_word_distances) / total_between
        else:
            avg_within_word_distance, avg_between_word_distance = 0, 0
    else:
        avg_distance, stdev_distance, avg_within_word_distance, avg_between_word_distance = 0, 0, 0, 0

    # Part 2: BLOBS
    # create list with a tuple (area, perimeter) for each blob
    features = map(lambda blob: (len(blob), find_blob_perimeter(blob)), blobs)
    # create list with a tuple (area, perimeter, shape_factor, roundness)
    features = list(map(lambda ft: (ft[0], ft[1], 4 * ft[0] * np.pi / ft[1] ** 2, ft[1] ** 2 / ft[0]), features))
    # Calculate averages
    total_blobs = len(features)
    if total_blobs > 0:
        avg_area = sum(area for area, perimeter, shape_factor, roundness in features) / total_blobs
        avg_perimeter = sum(perimeter for area, perimeter, shape_factor, roundness in features) / total_blobs
        avg_shape_factor = sum(shape_factor for area, perimeter, shape_factor, roundness in features) / total_blobs
        avg_roundness = sum(roundness for area, perimeter, shape_factor, roundness in features) / total_blobs
    else:
        avg_area = 0
        avg_perimeter = 0
        avg_shape_factor = 0
        avg_roundness = 0

    # Return all features of both components (part 1) and blobs (part 2)
    return avg_distance, stdev_distance, avg_within_word_distance, avg_between_word_distance, \
           avg_area, avg_perimeter, avg_shape_factor, avg_roundness

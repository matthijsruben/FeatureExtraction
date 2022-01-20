import IAMloader
import functools
import time
import numpy as np
import os
import statistics
import math


# GENERAL FUNCTIONS ----------------------------------------------------------------------------------------------------


def black_white_pixel(pixel, bits, slack):
    return 0 if pixel < (2**bits + slack) / 2 else 255


def black_white_row(row, slack):
    return list(map(lambda x: black_white_pixel(x, 8, slack), row))


def black_white_image(img, slack):
    return list(map(lambda x: black_white_row(x, slack), img))


def count_black_row(row, bits, slack):
    count = 0
    for pixel in row:
        if pixel < (2**bits + slack) / 2:
            count += 1
    return count


def count_black_white_transitions(row):
    transitions = 0
    previous = black_white_pixel(row[0], 8, 150)
    for pixel in row:
        current = black_white_pixel(pixel, 8, 150)
        if current != previous:
            transitions += 1
        previous = current
    return transitions


def has_white_neighbor(x, y, image, connectivity):
    if connectivity == 8:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= y+dy < len(image) and 0 <= x+dx < len(image[0]) and image[y+dy][x+dx] != 0 \
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


def find_border_pixels(width, height):
    border_pixels = []
    for x in range(width):
        border_pixels.append((x, 0))
        border_pixels.append((x, height - 1))
    for y in range(height):
        border_pixels.append((0, y))
        border_pixels.append((width - 1, y))
    return border_pixels


# WRITING ZONES --------------------------------------------------------------------------------------------------------


def ideal_hist(height, ub, lb, total, threshold):
    """
    :param height:      Height of the image. Integer value
    :param ub:          Upper baseline. Integer value.
    :param lb:          Lower baseline. Integer value.
    :param total:       Total amount of black pixels in the image. Integer value.
    :param threshold:   Percentage used to calculate amount of pixels in each zone. float between 0 and 1.
    :return:            an ideal histogram as a triple, representing the average amount of pixels for the lower zone,
                        middle zone, and upper zone, respectively
    For an 'ideal histogram', the following is assumed:
    - threshold% of the total amount of pixels is between ub and lb.
    - (100-threshold)/2% is between top line (height) and ub
    - (100-threshold)/2% is between bottom line (0) and lb
    """
    middle_zone = int(round(threshold * total))
    upper_lower_zone = int(round(((1-threshold)/2) * total))

    # amount of pixels per row in each zone
    middle = int(round(middle_zone / (ub - lb)))
    upper = int(round(upper_lower_zone / (height - ub)))
    lower = int(round(upper_lower_zone / lb))
    return lower, middle, upper


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
    error1 = sum(map(lambda x: (x - lower_ideal)**2, lower_img))
    error2 = sum(map(lambda x: (x - middle_ideal) ** 2, middle_img))
    error3 = sum(map(lambda x: (x - upper_ideal) ** 2, upper_img))
    return error1 + error2 + error3


def find_minimum_error(pixels_per_row):
    height = len(pixels_per_row)
    minimum = (-1, -1, np.inf)
    for ub in range(1, height):
        for lb in range(1, ub):
            err = hist_error(ub, lb, pixels_per_row)
            if err < minimum[2]:
                minimum = (ub, lb, err)
    return minimum


def writing_zones(image):
    black_white = list(map(lambda x: count_black_row(x, 8, 150), image))
    ub, lb, err = find_minimum_error(black_white)
    height = len(image)
    f1 = (height - ub) / height  # upper writing zone as a fraction of the total writing zone
    f2 = (ub - lb) / height  # middle zone as a fraction of the total writing zone
    f3 = lb / height  # lower zone as a fraction of the total writing zone
    return f1, f2, f3


# MEDIAN WIDTH ---------------------------------------------------------------------------------------------------------


def median_width(image):
    transitions = list(map(count_black_white_transitions, image))
    row_most_transitions = image[transitions.index(max(transitions))]
    prev = False
    dist = 0
    distances = []
    for pixel in row_most_transitions:
        white = black_white_pixel(pixel, 8, 150) == 255
        if white and (dist == 0 or (dist > 0 and prev)):
            dist += 1
        # Exclude whitespace after the last black pixel
        elif not white and prev:
            distances.append(dist)
            dist = 0
        prev = white
    # Exclude whitespace before the first black pixel
    if black_white_pixel(row_most_transitions[0], 8, 150) == 255:
        distances.pop(0)
    return statistics.median(distances)


# SLANTNESS ------------------------------------------------------------------------------------------------------------


def find_line(angle, length):
    line = set()
    for i in range(0, length + 1):
        vertical = i * math.sin(angle)
        horizontal = i * math.cos(angle)
        pixel = (round(horizontal), round(vertical))
        line.add(pixel)
    return sorted(list(line))


def slantness(image):
    black_white = black_white_image(image, 150)
    amount_of_angles = 40
    length = len(black_white)
    score = {}
    # Loop through all angles
    for i in range(1, amount_of_angles):
        angle = i * (math.pi / amount_of_angles)
        line = find_line(angle, length)
        score[angle] = 0
        # For each angle, loop through all shifts (shift 10 pixels everytime)
        for shift in range(0, len(image[0])):
            shifted_line = list(map(lambda pix: (pix[0] + shift, pix[1]), line))
            filtered_line = list(filter(lambda pix: 0 <= pix[0] < len(black_white[0]) and 0 <= pix[1] < len(black_white), shifted_line))
            total = len(filtered_line)
            img_line = list(map(lambda pix: black_white[pix[1]][pix[0]], filtered_line))
            # Calculate # transitions and # black pixels on the line
            transitions = count_black_white_transitions(img_line)
            blacks = len(list(filter(lambda pix: pix == 0, img_line)))
            # Only lines (> 3/4 * height of image); consisting of >=50% black pixels, with 1 consecutive piece of black
            if transitions <= 2 and blacks / total >= 0.5 and total >= 3 * len(black_white) / 4:
                score[angle] += 1
    sample = functools.reduce(lambda a, b: a + b, list(map(lambda x: [x[0]] * x[1], score.items())))
    stdev_angle = round(math.degrees(statistics.stdev(sample)), 1) if len(sample) > 1 else 0
    avg_angle = round(180 - math.degrees(statistics.mean(sample)), 1) if len(sample) >= 1 else 0
    max_angle = round(180 - math.degrees(max(score, key=score.get)), 1)
    return max_angle, avg_angle, stdev_angle


# COMPONENTS AND BLOBS -------------------------------------------------------------------------------------------------


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
                if is_blob and len(white_area) >= minimum_size and not any([pixel in border_pixels for pixel in white_area]):
                    blobs.append(white_area)
    return blobs


def find_components_and_blobs(image):
    black_white = black_white_image(image, 200)
    border_pixels = find_border_pixels(len(black_white[0]), len(black_white))
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
    features = list(map(lambda ft: (ft[0], ft[1], 4 * ft[0] * np.pi / ft[1]**2, ft[1]**2 / ft[0]), features))
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


# FEATURE EXTRACTION ---------------------------------------------------------------------------------------------------


def write_features_to_file(features, filename):
    # Check if there exists a file called 'features.txt' already in the same folder
    start = time.time()
    if os.path.isfile(filename):
        # Read file
        with open(filename, 'r') as infile:
            lines = infile.readlines()
            # Append Feature data, in case the file contains information already
            if len(lines) == len(features):
                if features[0] in lines[0]:
                    print("The feature \'" + features[0] + "\' is already mentioned in the file \'" + filename + "\'")
                    return
                print("Appending data of the feature \'" + features[0] +
                      "\' to the already existing file \'" + filename + "\'...")
                content = ''.join(map(lambda x: x[0].rstrip('\n') + ',' + str(x[1]) + '\n', zip(lines, features)))
                with open(filename, 'w') as outfile:
                    outfile.write(content)
            # Write Feature data, in case the file is empty
            elif len(lines) == 0:
                print("Writing data of the feature \'" + features[0] +
                      "\' to the already existing, but empty, file \'" + filename + "\'...")
                content = ''.join([str(feature) + '\n' for feature in features])
                with open(filename, 'w') as outfile:
                    outfile.write(content)
            # Don't write anything to the file if there are not as many new features as lines in the current text file
            else:
                print("The amount of features is not the same as the amount of lines in the file \'" + filename + "\'")
    else:
        print("Creating a file called \'" + filename + "\' and writing data of the feature \'" + features[0] +
              "\' to it...")
        content = ''.join([str(feature) + '\n' for feature in features])
        # Write file
        with open(filename, 'w') as outfile:
            outfile.write(content)
    end = time.time()
    print("Finished! Took ", round(end - start, 3), " seconds.")


def feature_extraction(data, until, filename, feature_func, feature_names):
    features = [[name] for name in feature_names]
    try:
        if len(features) != len(feature_func(data[0][1])):
            print("The function does not return the same amount of features as the amount of feature-names that are "
                  "passed as argument")
            return
    except TypeError:
        if len(feature_names) != 1:
            print("The function does not return the same amount of features as the amount of feature-names that are "
                  "passed as argument")
            return
    start = time.time()
    for datapoint in data[0:until]:
        image = datapoint[1]
        if len(feature_names) == 1:
            value = feature_func(image)
            features[0].append(value)
        else:
            values = feature_func(image)
            for i in range(len(features)):
                features[i].append(values[i])
    end = time.time()
    print("Feature extraction time in oder to calculate the feature(s) " + ", ".join(feature_names)
          + " of " + str(until) + " images: " + str(round(end - start, 3)) + " seconds.")
    for elem in features:
        write_features_to_file(elem, filename)


# DATA PREPROCESSING ---------------------------------------------------------------------------------------------------
# Pre process data to be in format [(writer, 2d-image), ...]
data = IAMloader.load_data('Data/lines/lines.tgz')
data = map(lambda x: [(x[0], img) for img in x[1]], data.items())
data = functools.reduce(lambda a, b: a + b, data)
print("Finished pre-processing!")

# ACTUAL FEATURE EXTRACTION --------------------------------------------------------------------------------------------
# Do feature extraction and write features to a file called 'features.txt'
filename = 'features.txt'
feature_extraction(data, 100, filename, writing_zones, ['upper_zone', 'middle_zone', 'lower_zone'])  # ~1 min
feature_extraction(data, 100, filename, median_width, ['median_width'])  # ~1 min
feature_extraction(data, 100, filename, slantness, ['most_occurring_angle', 'average_angle', 'stdev_angle'])  # ~ 10 min
feature_extraction(data, 100, filename, features_components_blobs, ['comp_dist', 'stdev_comp_dist', 'within_word_dist', 'between_word_dist', 'blob_area', 'blob_perimeter', 'blob_shape_factor', 'blob_roundness'])  # ~ 50 min
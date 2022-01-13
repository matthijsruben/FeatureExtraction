import functools
import operator
import os
import statistics
import sys
import time

import math
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

import IAMloader


def black_white_pixel(pixel, bits, slack):
    return 0 if pixel < (2 ** bits + slack) / 2 else 255


def black_white_row(row, slack):
    return list(map(lambda x: black_white_pixel(x, 8, slack), row))


def black_white_image(img, slack):
    return list(map(lambda x: black_white_row(x, slack), img))


def count_black_row(row, bits, slack):
    count = 0
    for pixel in row:
        if pixel < (2 ** bits + slack) / 2:
            count += 1
    return count


def count_black_row_2(row, bits, slack):
    return len(list(filter(lambda pix: pix < (2 ** bits + slack) / 2, row)))


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
    upper_lower_zone = int(round(((1 - threshold) / 2) * total))

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
    error1 = sum(map(lambda x: (x - lower_ideal) ** 2, lower_img))
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


def draw_baselines(ub, lb, image):
    new_image = image
    new_image[ub] = np.array(new_image[ub])
    new_image[ub].fill(0)
    new_image[lb] = np.array(new_image[lb])
    new_image[lb].fill(0)
    return new_image


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


def writing_zones(image):
    black_white = list(map(lambda x: count_black_row(x, 8, 150), image))
    ub, lb, err = find_minimum_error(black_white)
    height = len(image)
    f1 = (height - ub) / height  # upper writing zone as a fraction of the total writing zone
    f2 = (ub - lb) / height  # middle zone as a fraction of the total writing zone
    f3 = lb / height  # lower zone as a fraction of the total writing zone
    return f1, f2, f3


def count_black_white_transitions(row):
    transitions = 0
    previous = black_white_pixel(row[0], 8, 150)
    for pixel in row:
        current = black_white_pixel(pixel, 8, 150)
        if current != previous:
            transitions += 1
        previous = current
    return transitions


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


def find_contour(image):
    black_white = black_white_image(image, 150)
    new_image = [[-1 for pix in row] for row in black_white]
    for y in range(0, len(black_white)):
        for x in range(0, len(black_white[y])):
            if black_white[y][x] == 0 and has_white_neighbor(x, y, black_white, 4):
                new_image[y][x] = 0
            else:
                new_image[y][x] = 255
    return new_image


def connect_component_contour(image, x, y, visited):
    # Mark the current pixel as visited if it is a black pixel
    visited.append((x, y))

    # Repeat for all neighbors to this pixel (NOTE 8-connectivity only works for contours!)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if 0 <= y + dy < len(image) and 0 <= x + dx < len(image[0]) and \
                    image[y + dy][x + dx] == 0 and (x + dx, y + dy) not in visited:
                visited = connect_component_contour(image, x + dx, y + dy, visited)
    return visited


def find_connected_components_contour(image):
    contours = find_contour(image)
    components = {}
    c = 1
    visited = []
    for y in range(0, len(contours)):
        for x in range(0, len(contours[0])):
            if contours[y][x] == 0 and (x, y) not in visited:
                # find connected component trough recursion
                component = connect_component_contour(contours, x, y, [])
                visited.extend(component)
                for pixel in component:
                    components[pixel] = c
                c += 1
            elif contours[y][x] == 255:
                components[(x, y)] = 0
                visited.append((x, y))
    return components, c


def connect_component(image, x, y, visited):
    # Mark the current pixel as visited
    visited.append((x, y))

    # Repeat for all neighbors to this pixel (NOTE 4-connectivity)
    neighbors = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
    for nb in neighbors:
        if 0 <= nb[1] < len(image) and 0 <= nb[0] < len(image[0]) and image[nb[1]][nb[0]] == image[y][x] \
                and nb not in visited:
            visited = connect_component(image, nb[0], nb[1], visited)
    return visited


def find_connected_components(image):
    sys.setrecursionlimit(200 * 2000)  # 200 * 2000 is the average image size
    black_white = black_white_image(image, 150)
    components = {}
    c = 1
    visited = []
    for y in range(0, len(black_white)):
        for x in range(0, len(black_white[0])):
            if (x, y) not in visited:
                # find connected component trough recursion
                component = connect_component(black_white, x, y, [])
                visited.extend(component)
                for pixel in component:
                    components[pixel] = c
                c += 1
    return components, c


def find_border_pixels(width, height):
    border_pixels = []
    for x in range(width):
        border_pixels.append((x, 0))
        border_pixels.append((x, height - 1))
    for y in range(height):
        border_pixels.append((0, y))
        border_pixels.append((width - 1, y))
    return border_pixels


def find_blobs(image):
    components, count = find_connected_components(image)
    border_classes = set()
    border_pixels = find_border_pixels(len(image[0]), len(image))
    for pixel, cl in components.items():
        if pixel in border_pixels:
            border_classes.add(cl)
    # Remove all pixels that belong to border classes and remove all pixels that are black
    blobs = dict(
        filter(lambda item: item[1] not in border_classes and image[item[0][1]][item[0][0]] != 0, components.items()))
    count -= len(border_classes) - 1  # all classes that are border classes should be removed from the count
    return blobs, count


def find_line_recursion(angle, line, length, pixel, end, startpoint):
    line.append(pixel)
    if pixel != end and pixel[0] != end[0] and pixel[1] != end[1]:
        opposite_angle = (math.pi / 2) - angle
        dx = length * math.tan(opposite_angle)
        movement = round(startpoint + dx, 3)
        if movement < 1:
            line = find_line_recursion(angle, line, 1, (pixel[0], pixel[1] + 1), end, movement)
        elif movement == 1:
            line = find_line_recursion(angle, line, 1, (pixel[0] + 1, pixel[1] + 1), end, 0)
        else:
            remaining = 1 - startpoint
            used_length = remaining * math.tan(angle)
            new_length = length - used_length
            line = find_line_recursion(angle, line, new_length, (pixel[0] + 1, pixel[1]), end, 0)
    return line


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
            filtered_line = list(
                filter(lambda pix: 0 <= pix[0] < len(black_white[0]) and 0 <= pix[1] < len(black_white), shifted_line))
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


def extract_column(image, col_index):
    """
    :param image: the image to extract the column from.
    :param col_index: the column index.
    :return: The array of pixels in the column at the given index.
    """
    return [image[y][col_index] for y in range(0, len(image))]


def find_upper_boundary_pixel(column):
    """
    :param column: the array of pixels in the column
    :return: The index of the uppermost black pixel in the provided column; None if no black pixels exist in this
    column.
    """
    for y in range(0, len(column)):
        if column[y] == 0:
            return y

    return None


def find_lower_boundary_pixel(column):
    """
    :param column: the array of pixels in the column
    :return: The index of the lowermost black pixel in the provided column; None if no black pixels exist in this
    column.
    """
    for y in range(len(column) - 1, -1, -1):
        if column[y] == 0:
            return y

    return None


def find_characteristic_contour(image, lower):
    """
    Returns the characteristic contour of the provided image after elimination of discontinuities in y-direction, in a
    normalized format such that the returned values are always >= 0.
    :param image: the image to find the contours for.
    :param lower: whether to extract the lower contour (True) or the upper contour (False).
    :return: The normalized contour values.
    """
    bw_image = black_white_image(image, 150)

    contours = []
    last_contour_y = None

    for x in range(0, len(bw_image[0])):
        column = extract_column(bw_image, x)
        contour_y = find_lower_boundary_pixel(column) if lower else find_upper_boundary_pixel(column)

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


def find_characteristic_contour_polynomial(contour):
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


def find_local_extremes(contour, maxima=True, window=20):
    """
    Finds the x-coordinates of the local extremes, defined as points on the characteristic contour such that there is no
    other point within a neighbourhood of the given size that has a larger (smaller) y-value.
    :param contour: the contours to find extremes in.
    :param maxima: whether to find the maxima (True) or the minima (False)
    :param window: the size of the neighborhood in which the point has to be the maximum (minimum)
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


# SHOW STUFF -----------------------------------------------------------------------------------------------------------


def visualize_contour(contour):
    new_image = [[255 for pix in range(0, len(contour))] for row in range(0, max(contour) + 1)]
    for x, y in enumerate(contour):
        new_image[y][x] = 0

    plt.imshow(new_image)


def visualize_contour_extremes(contour):
    maxima = find_local_extremes(contour, True)
    maxima_y = [contour[x] for x in maxima]
    minima = find_local_extremes(contour, False)
    minima_y = [contour[x] for x in minima]

    markerStyle = "x"
    markerSize = 20
    plt.scatter(maxima, maxima_y, s=markerSize, marker=markerStyle)
    plt.scatter(minima, minima_y, s=markerSize, marker=markerStyle)


def visualize_contour_slant(contour):
    (intersection, slope, mse) = find_characteristic_contour_polynomial(contour)

    X = np.arange(0, len(contour), 1)
    Y = intersection + slope * X

    plt.plot(X, Y)


def visualize_contour_features(image):
    bw_image = black_white_image(image, 150)
    lower_contour = find_characteristic_contour(bw_image, True)

    plt.subplot(3, 1, 1)
    plt.imshow(bw_image)

    plt.subplot(3, 1, 2)
    visualize_contour(lower_contour)
    visualize_contour_extremes(lower_contour)
    visualize_contour_slant(lower_contour)

    upper_contour = find_characteristic_contour(bw_image, False)
    plt.subplot(3, 1, 3)
    visualize_contour(upper_contour)
    visualize_contour_extremes(upper_contour)
    visualize_contour_slant(upper_contour)

    plt.show()

# for i in range(0, 3):
#     image = data[i*10][1]
#     IAMloader.show(image)
#
#     # SHOW IMAGE IN BINARY FORM (ONLY BLACK OR WHITE PIXELS)
#     new_image = black_white_image(image, 150)
#     IAMloader.show(image)
#     IAMloader.show(new_image)
#
#     # SHOW UPPER BASELINE AND LOWER BASELINE
#     black_white = list(map(lambda x: count_black_row(x, 8, 150), image))
#     ub, lb, err = find_minimum_error(black_white)
#     new_image = image
#     new_image[ub].fill(0)
#     new_image[lb].fill(0)
#     IAMloader.show(new_image)
#
#     # SHOW CONTOUR
#     contour_image = find_contour(image)
#     IAMloader.show(image)
#     IAMloader.show(contour_image)
#
#     # SHOW CONNECTED COMPONENTS (VERY SLOW)
#     # image = image[0:80]
#     # image = list(map(lambda row: row[60:160], image))
#     # components, c = find_connected_components_contour(image)
#     components, c = find_connected_components(image)
#     component_image = [[255 for pix in row] for row in image]
#     for y in range(0, len(image)):
#       for x in range(0, len(image[0])):
#           if components[(x, y)] != 0:
#               component_image[y][x] = components[(x, y)] * int(255 / c)
#     IAMloader.show(component_image)
#
#     # SHOW BLOBS (VERY SLOW)
#     # image = image[0:80]
#     # image = list(map(lambda row: row[60:160], image))
#     blobs, blobs_c = find_blobs(image)
#     print(blobs)
#     print(blobs_c)
#     blobs_image = [[255 for pix in row] for row in image]
#     for y in range(0, len(image)):
#       for x in range(0, len(image[0])):
#           if (x, y) in blobs.keys():
#               blobs_image[y][x] = blobs[(x, y)] * int(255 / blobs_c)
#     IAMloader.show(blobs_image)
#
#     # SHOW SLANTNESS OF IMAGE
#     schuinheid = slantness(image)
#     print("Hoek is", schuinheid[0], "graden")
#     black_white = black_white_image(image, 150)
#     line = find_line(math.radians(180 - schuinheid), len(black_white))
#     for shift in range(0, len(image[0]), 50):
#         shifted_line = list(map(lambda pix: (pix[0] + shift, pix[1]), line))
#         for pixel in shifted_line:
#             if pixel[1] < len(black_white) and pixel[0] < len(black_white[0]):
#                 black_white[pixel[1]][pixel[0]] = 150
#     IAMloader.show(black_white)

# TEST STUFF -----------------------------------------------------------------------------------------------------------
# line1 = find_line_recursion(angle, [], 1, (0, 0), (round(length * math.cos(angle)), round(length * math.sin(angle))), 0)

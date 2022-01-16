import sys

import math

from src import utils


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
    black_white = utils.get_bw_image(image, 150)
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
    black_white = utils.get_bw_image(image, 150)
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

# for i in range(0, 3):
#     image = data[i*10][1]
#     IAMloader.show(image)
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

# TEST STUFF -----------------------------------------------------------------------------------------------------------
# line1 = find_line_recursion(angle, [], 1, (0, 0), (round(length * math.cos(angle)), round(length * math.sin(angle))), 0)

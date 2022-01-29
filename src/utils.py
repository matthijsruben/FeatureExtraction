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


def get_bw_pixel(pixel, bits, slack):
    return 0 if pixel < (2 ** bits + slack) / 2 else 255


def get_bw_row(row, slack):
    return list(map(lambda x: get_bw_pixel(x, 8, slack), row))


def get_border_pixels(width, height):
    border_pixels = []
    for x in range(width):
        border_pixels.append((x, 0))
        border_pixels.append((x, height - 1))
    for y in range(height):
        border_pixels.append((0, y))
        border_pixels.append((width - 1, y))
    return border_pixels

def get_bw_image(image, slack):
    """
    Returns a black-and-white (i.e. binary) version of the provided image.
    :param image: The image to get a black
    :param slack:
    :return:
    """
    return list(map(lambda x: get_bw_row(x, slack), image))


def get_bw_transition_count(row):
    transitions = 0
    previous = get_bw_pixel(row[0], 8, 150)

    for pixel in row:
        current = get_bw_pixel(pixel, 8, 150)

        if current != previous:
            transitions += 1

        previous = current

    return transitions

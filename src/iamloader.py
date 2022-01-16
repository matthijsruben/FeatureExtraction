import functools
import logging
import tarfile

import cv2
import numpy as np


def load_data(path):
    """
    Returns a dictionary of the images in the provided tar-file, where:
    key: writer (string)
    val: a list of images (array)
    Each image is a two-dimensional array with grayscale pixel-values in the range 0-255.
    :param path: the path to the tar-file
    :return: A writer-keyed dictionary of images.
    """
    logging.info("Loading data from path: '{}'".format(path))

    data = {}
    with tarfile.open(path, 'r:gz') as tar:
        for member, name in zip(tar.getmembers(), tar.getnames()):
            f = tar.extractfile(member)

            if f is not None:
                content = f.read()
                f.close()
                """
                Useful flags for decoding:
                -1  :   Return the loaded image as is (unchanged)
                0   :   Convert image to single channel grayscale image
                16  :   Convert image to single channel grayscale image and image size reduced 1/2
                32  :   Convert image to single channel grayscale image and image size reduced 1/4
                64  :   Convert image to single channel grayscale image and image size reduced 1/8
                """
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), -1)

                # add key and value to data dict
                writer = name.split('/')[1]

                if writer not in data.keys():
                    data[writer] = []

                data[writer].append(image)
    return data


def load_data_as_tuples(path):
    """
    :param path: the path to the tar-file containing the images
    :return: An array of (writer, image) tuples.
    """
    data = load_data(path)
    data = map(lambda x: [(x[0], img) for img in x[1]], data.items())
    data = functools.reduce(lambda a, b: a + b, data)

    return data

import functools
import logging
import tarfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def load_data(lines_path, metadata_path):
    """
    Returns a dictionary of the images in the provided tar-file, where:
    key: writer (string)
    val: a list of images (array)
    Each image is a two-dimensional array with grayscale pixel-values in the range 0-255.
    :param lines_path: the path to the tar-file
    :param metadata_path: the path to the folder containing the metadata (no / on end)
    :return: A writer-keyed dictionary of (filename, image) tuples.
    """
    logging.info("Loading data from path: '{}'".format(lines_path))

    data = {}
    with tarfile.open(lines_path, 'r:gz') as lines_tar:
        for member, name in zip(lines_tar.getmembers(), lines_tar.getnames()):
            f = lines_tar.extractfile(member)
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

                filename = name.split('/')[-1]
                document_identifier = name.split('/')[-2]

                # Obtain the metadata for this document. This could be extracted from the image file itself,
                # but that seems more cumbersome.
                metadata_file = open("{}/{}.xml".format(metadata_path, document_identifier))
                writer = ET.parse(metadata_file).getroot().attrib["writer-id"]
                metadata_file.close()

                if writer not in data.keys():
                    data[writer] = []

                logging.debug("Appending {} from writer {}".format(filename, writer))
                data[writer].append((filename, image))
        return data


def load_data_as_tuples(lines_path, metadata_path):
    """
    :param lines_path: the path to the tar-file containing the images
    :param metadata_path: the path to the tar-file containing the metadata
    :return: An array of (writer, image) tuples.
    """
    data = load_data(lines_path, metadata_path)
    data = map(lambda x: [(x[0], image) for image in x[1]], data.items())
    data = functools.reduce(lambda a, b: a + b, data)

    return data

import concurrent.futures
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from src.preparation.features.contour import get_contour_features
from src.preparation.features.medianwidth import get_median_width_features
from src.preparation.features.slantness import get_slantness_features
from src.preparation.features.writingzones import get_writing_zone_features
from src.preparation.iamloader import load_data_as_tuples
from src.preparation.models import TextImage

DEFAULT_OUTPUT_DIR = "./output/features"


def write_or_append_features(images: [TextImage], path):
    """
    Writes the features associated with the provided images to a CSV-file at the given path. If a CSV-file exists at the
    path, it appends the features (without headers). Else, it creates a new file with column headers.
    :param images: the images with features extracted
    :param path: the filepath to write the features to
    """
    df = pd.DataFrame([image.as_dict() for image in images])

    if not os.path.isfile(path):
        logging.info("Writing {} features to {}".format(len(images), path))
        df.to_csv(path, index=False)
    else:
        logging.info("Appending {} features to {}".format(len(images), path))
        df.to_csv(path, mode='a', index=False, header=False)


def write_features(images: [TextImage], path):
    """
    Writes the features associated with the provided images to a CSV-file at the given path.

    If a file at the given path already exists, the number of provided images must match the number of images for which
    the file at the given path already contains features, as the columns are then simply appended (the same starting
    point and order is assumed). Otherwise. the file at the path will first need to be deleted.
    :param images: the images with features extracted
    :param path: the filepath to write the features to
    """
    # TODO: Implement write_features with smarter merging such that features are added based on the file names in the CSV.
    logging.info("Writing features for {} images to path '{}'".format(len(images), path))
    new_df = pd.DataFrame([image.as_dict() for image in images])

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        # No previous export was found, so the new features can be written immediately without further checks
        logging.info("File at path '{}' does not yet exist; creating new file".format(path))
        new_df.to_csv(path, index=False)
        logging.info("Features for {} images written to file '{}'".format(len(images), path))
        return

    logging.info("File at path '{}' already exists.".format(path))
    if new_df.shape[0] != df.shape[0]:
        logging.error("The number of images to write features for must match the number of images already in the "
                      "CSV-file.")
        raise Exception("The number of images to write features must match the number of images already in the "
                        "CSV-file")

    new_columns = np.setdiff1d(new_df.columns.values, df.columns.values)
    if new_columns.size == 0:
        logging.info("All columns already exist in the file. Returning...")
        return

    logging.info("Extracted new features. Adding the following new columns: {}".format(", ".join(new_columns)))

    for column_name in new_columns:
        df[column_name] = new_df[column_name]

    df.to_csv(path, index=False)
    logging.info("Features for {} images written to file '{}'".format(len(images), path))


def extract_features(datapoint):
    """
    Extracts a number of features for the provided datapoint and returns a TextImage with the related metadata set.
    :param datapoint: the datapoint in the format (writer, (filename, image))
    :return: a TextImage with the related metadata (writer, filename and features) set.
    """
    writer = datapoint[0]
    filename = datapoint[1][0]
    image = datapoint[1][1]

    logging.debug("Processing image {}".format(filename))

    text_image = TextImage(writer, filename, image)
    text_image.add_features(get_writing_zone_features(text_image))
    text_image.add_features(get_median_width_features(text_image))
    lc_features, uc_features = get_contour_features(text_image)
    text_image.add_features(lc_features)
    text_image.add_features(uc_features)
    text_image.add_features(get_slantness_features(text_image))

    return text_image


def extract_and_write_features(output_dir_abs):
    """
    Loads the IAM handwriting database from the file system and extracts metadata as well as features from the lines
    of text.
    :param output_dir_abs: The directory to write the resulting .csv-file to.
    """
    data = load_data_as_tuples('data/lines/lines.tgz', 'data/lines/xml')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = "{}/{}_features.csv".format(output_dir_abs, timestamp)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        images = []

        for datapoint in data[:10]:
            futures.append(executor.submit(extract_features, datapoint))
        for future in concurrent.futures.as_completed(futures):
            images.append(future.result())

            if (len(images) % 10) == 0:
                # Periodically write features to prevent data loss when processing fails late in the process
                write_or_append_features(images, output_path)
                images.clear()

        write_or_append_features(images, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) == 1:
        logging.info("Script ran without arguments")
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        logging.info("Script ran with arguments: {}".format(" ".join(sys.argv)))
        output_dir = sys.argv[1].strip()

    if output_dir.endswith("/"):
        # Remove leading slash
        output_dir = output_dir[-1]

    logging.info("Configured to output to directory: {}".format(output_dir))
    output_dir = os.path.abspath(output_dir)
    logging.info("Absolute path of output directory: {}".format(output_dir))

    if not os.path.isdir(output_dir):
        logging.info("Directory {} does not yet exist. Creating...".format(output_dir))
        os.makedirs(output_dir, exist_ok=True)

    extract_and_write_features(output_dir)

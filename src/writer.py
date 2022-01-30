import logging
import os

import numpy as np
import pandas as pd

from src.models import TextImage

path = "./features.txt"


def write_or_append_features(images: [TextImage], path):
    """
    Writes the features associated with the provided images to a CSV-file at the given path. If a CSV-file exists at the
    path, it appends the features (without headers). Else, it creates a new file with column headers.
    :param images: the images with features extracted
    :param path: the filepath to write the features to
    :return:
    """
    df = pd.DataFrame([image.as_dict() for image in images])

    if not os.path.isfile(path):
        logging.info("Writing {} features to {}".format(len(images), path))
        df.to_csv(path, index=False)
    else:
        logging.info("Appending {} features to {}".format(len(images), path))
        df.to_csv(path, mode='a', index=False, header=False)


# TODO: Implement write_features with smarter merging such that features are added based on the file names in the CSV.
def write_features(images: [TextImage], path):
    """
    Writes the features associated with the provided images to a CSV-file at the given path.

    If a file at the given path already exists, the number of provided images must match the number of images for which
    the file at the given path already contains features, as the columns are then simply appended (the same starting
    point and order is assumed). Otherwise. the file at the path will first need to be deleted.
    :param images: the images with features extracted
    :param path: the filepath to write the features to
    """
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

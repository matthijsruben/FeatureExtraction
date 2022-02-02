import logging
import os
import sys

import pandas as pd

from src.preparation.loader import load_datasets

DEFAULT_FEATURES_FILE = "./output/features/retain/features.csv"


def get_type_from_line(mapping, document):
    for (data_type, lines) in mapping:
        if document in lines:
            return data_type

    return "unknown"


def get_line_mapping():
    with open("./data/task/testset.txt", "r") as test_file, \
            open("./data/task/trainset.txt", "r") as train_file, \
            open("./data/task/validationset1.txt", "r") as validation_1_file, \
            open("./data/task/validationset2.txt", "r") as validation_2_file:
        test = []
        for line in test_file:
            test.append(line.rstrip())

        train = []
        for line in train_file:
            train.append(line.rstrip())

        validation_1 = []
        for line in validation_1_file:
            validation_1.append(line.rstrip())

        validation_2 = []
        for line in validation_2_file:
            validation_2.append(line.rstrip())

    return [
        ("test", test),
        ("train", train),
        ("validation_1", validation_1),
        ("validation_2", validation_2)
    ]


def get_categorised_data(path):
    data = pd.read_csv(path)

    suffix_length = len(".png")
    line_id = data.filename.map(lambda filename: filename[:-suffix_length])
    data.insert(data.columns.get_loc("filename") + 1, "line_id", line_id)

    mapping = get_line_mapping()
    data_type = data.line_id.map(lambda line_id: get_type_from_line(mapping, line_id))
    data.insert(data.columns.get_loc("line_id") + 1, "type", data_type)

    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) == 1:
        logging.info("Script ran without arguments")
        features_file = DEFAULT_FEATURES_FILE
    else:
        logging.info("Script ran with arguments: {}".format(" ".join(sys.argv)))
        features_file = sys.argv[1].strip()

    logging.info("Configured to read from file: {}".format(features_file))
    features_file = os.path.abspath(features_file)
    logging.info("Absolute path of file: {}".format(features_file))
    if not os.path.isfile(features_file):
        raise FileNotFoundError("No file exists at the defined path: {}".format(features_file))

    categorised_data = get_categorised_data(features_file)
    categorised_data.to_csv("./output/features/retain/features_categorised.csv", index=False)

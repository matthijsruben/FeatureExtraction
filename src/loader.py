import pandas as pd


def load_datasets(path="./output/features/retain/features_categorised.csv"):
    """
    Loads the categorised feature set from the file system and splits it into a train, test, and two validation sets.
    :param path: the path to the categorised feature file
    :return: A tuple of (training_data, test_data, validation_data_1, validation_data_2)
    """
    df = pd.read_csv(path)

    train = df.query('type == "train"')
    test = df.query('type == "test"')
    validation_1 = df.query('type == "validation_1"')
    validation_2 = df.query('type == "validation_2"')

    return train, test, validation_1, validation_2

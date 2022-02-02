import pandas as pd

META_COLUMNS = ['writer', 'filename', 'line_id', 'type']
FAULTY_FEATURES = [
    'lc_max_slopes_left_avg', 'lc_max_slopes_right_avg', 'lc_min_slopes_left_avg', 'lc_min_slopes_right_avg',
    'uc_max_slopes_left_avg', 'uc_max_slopes_right_avg', 'uc_min_slopes_left_avg', 'uc_min_slopes_right_avg'
]


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

    return df, train, test, validation_1, validation_2


def load_features(without_features=None):
    if without_features is None:
        without_features = []
    drop_columns = META_COLUMNS + FAULTY_FEATURES + without_features

    data, train, test, validation_1, validation_2 = load_datasets()

    y_train = train.writer
    x_train = train.drop(drop_columns, axis=1).astype("float32")
    y_val_1 = validation_1.writer
    x_val_1 = validation_1.drop(drop_columns, axis=1).astype("float32")
    y_val_2 = validation_2.writer
    x_val_2 = validation_2.drop(drop_columns, axis=1).astype("float32")
    y_test = test.writer
    x_test = test.drop(drop_columns, axis=1).astype("float32")

    return data, (x_train, y_train), (x_val_1, y_val_1), (x_val_2, y_val_2), (x_test, y_test)

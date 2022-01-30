import random

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Normalization

from src.training import loss_functions
from src.training.loader import load_features
from src.training.statistics import generate_statistics, write_statistics


def data_generator(x, y, batch_size=128):
    x = x.values.tolist()
    y = y.values.tolist()

    while True:
        a = []
        p = []
        n = []
        for _ in range(batch_size):
            pos_neg = random.sample(list(set(y)), 2)
            positive_sample = random.sample([d for d, l in zip(x, y) if l == pos_neg[0]], 2)
            negative_sample = random.choice([d for d, l in zip(x, y) if l == pos_neg[1]])
            a.append(positive_sample[0])
            p.append(positive_sample[1])
            n.append(negative_sample)
        yield [np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32")


def generate_normalizer(x):
    normalizer = Normalization()
    normalizer.adapt(x)

    return normalizer


def generate_model(normalizer, nodes_per_layer, activation="relu"):
    dense_layers = []
    for nodes in nodes_per_layer:
        dense_layers.append(Dense(nodes, activation))

    return Sequential([normalizer] + dense_layers)


def generate_triplet_model(model, input_shape):
    triplet_model_a = Input(input_shape)
    triplet_model_p = Input(input_shape)
    triplet_model_n = Input(input_shape)
    triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])

    return Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)


def main():
    data, (x_train, y_train), (x_val_1, y_val_1), (x_val_2, y_val_2), (x_test, y_test) = load_features()

    input_neurons = x_train.shape[1]

    normalizer = generate_normalizer(x_train)
    model = generate_model(normalizer, [192, 416])

    triplet_model = generate_triplet_model(model, (input_neurons,))
    triplet_model.compile(loss=loss_functions.triplet_loss_euler, optimizer="adam")
    triplet_model.fit(data_generator(x_train, y_train), steps_per_epoch=200, epochs=20)

    embedding = triplet_model.layers[3]
    test_output = embedding.predict(x_test, verbose=1)

    stats = generate_statistics(range(1, 120), test_output, y_test, True)
    write_statistics(stats, "triplet_loss_euler")


if __name__ == "__main__":
    main()

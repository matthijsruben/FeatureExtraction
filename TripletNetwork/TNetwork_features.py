import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import umap.umap_ as umap
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

data = pd.read_csv('../features.csv').values.tolist()

y = [i[0] for i in data]
X = np.array([i[1:]for i in data])
classes = list(set(y))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# input_neurons = len(x_train[0])
input_neurons = x_train.shape[1]

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# print(y_train)
# y_train = LabelEncoder.fit_transform(y_train)
# print(y_train)


def data_generator(batch_size=100):
    while True:
        a = []
        p = []
        n = []
        for _ in range(batch_size):
            pos_neg = random.sample(classes, 2)
            positive_sample = random.sample([d for d, l in zip(x_train, y_train) if l == pos_neg[0]], 2)
            negative_sample = random.choice([d for d, l in zip(x_train, y_train) if l == pos_neg[1]])
            a.append(positive_sample[0])
            p.append(positive_sample[1])
            n.append(negative_sample)
        yield [np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32")


def triplet_loss(y_true, y_pred):
    anchor_out = y_pred[:, 0:23]
    positive_out = y_pred[:, 23:46]
    negative_out = y_pred[:, 46:69]

    # # pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    # # neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)
    #
    # probs = K.softmax([pos_dist, neg_dist], axis=0)
    #
    # return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

    pos_dist = K.sum((anchor_out - positive_out) ** 2)
    neg_dist = K.sum((anchor_out - negative_out) ** 2)

    delta_plus = K.pow(pos_dist, 2.7182828459) / (K.pow(pos_dist, 2.7182828459) + K.pow(neg_dist, 2.7182828459))
    delta_min = K.pow(neg_dist, 2.7182828459) / (K.pow(pos_dist, 2.7182828459) + K.pow(neg_dist, 2.7182828459))

    return K.sum(delta_plus + (1 - delta_min) ** 2)


# input_layer = Input((input_neurons, input_neurons, 1))
input_layer = Input(shape=(x_train.shape[1],))
x = Dense(input_neurons, activation='relu')(input_layer)
x = Dense(input_neurons, activation='relu')(x)
x = Dense(input_neurons, activation='relu')(x)
model = Model(input_layer, x)

model.summary()

triplet_model_a = Input((input_neurons,))
triplet_model_p = Input((input_neurons,))
triplet_model_n = Input((input_neurons,))
triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
triplet_model.summary()

triplet_model.compile(loss=triplet_loss, optimizer="adam")
# triplet_model.fit_generator(data_generator(), steps_per_epoch=150, epochs=3)
triplet_model.fit(data_generator(), steps_per_epoch=150, epochs=50)

model_embeddings = triplet_model.layers[3].predict(x_test, verbose=1)
print(model_embeddings.shape)

reduced_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="correlation").fit_transform(model_embeddings)
print(reduced_embeddings.shape)

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])#, c=y_test)

for i, txt in enumerate(y_test):
    plt.annotate(txt, (reduced_embeddings[:, 0][i], reduced_embeddings[:, 1][i]))
plt.show()

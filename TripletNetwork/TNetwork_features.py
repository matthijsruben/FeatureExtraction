import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

data = pd.read_csv('../features.csv').values.tolist()

y = [i[0] for i in data]
X = np.array([i[1:]for i in data])
classes = list(set(y))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

input_neurons = x_train.shape[1]

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")


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

    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)

    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

    # pos_dist = K.sum((anchor_out - positive_out) ** 2)
    # neg_dist = K.sum((anchor_out - negative_out) ** 2)
    #
    # delta_plus = K.pow(pos_dist, 2.7182828459) / (K.pow(pos_dist, 2.7182828459) + K.pow(neg_dist, 2.7182828459))
    # delta_min = K.pow(neg_dist, 2.7182828459) / (K.pow(pos_dist, 2.7182828459) + K.pow(neg_dist, 2.7182828459))
    #
    # delta_plus = K.pow(2.7182828459, pos_dist) / (K.pow(2.7182828459, pos_dist) + K.pow(2.7182828459, neg_dist))
    # delta_min = K.pow(2.7182828459, neg_dist) / (K.pow(2.7182828459, pos_dist) + K.pow(2.7182828459, neg_dist))
    #
    # return K.sum(delta_plus + (1 - delta_min) ** 2)

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
triplet_model.fit(data_generator(), steps_per_epoch=150, epochs=50)

model_embeddings = triplet_model.layers[3].predict(x_test, verbose=1)
print(model_embeddings.shape)

# reduced_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="correlation").fit_transform(model_embeddings)
# print(reduced_embeddings.shape)
#
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
#
# for i, txt in enumerate(y_test):
#     plt.annotate(txt, (reduced_embeddings[:, 0][i], reduced_embeddings[:, 1][i]))
# plt.show()

correct_1, wrong_1 = 0, 0
correct_2, wrong_2 = 0, 0
correct_5, wrong_5 = 0, 0
correct_10, wrong_10 = 0, 0
for i in range(len(model_embeddings)):
    for j in range(len(model_embeddings)):
        similarity = K.sqrt(K.sum(K.square(model_embeddings[i] - model_embeddings[j]), axis=-1))
        if y_test[i] == y_test[j]:
            if similarity <= 1:
                correct_1 += 1
            else:
                wrong_1 += 1
            if similarity <= 2:
                correct_2 += 1
            else:
                wrong_2 += 1
            if similarity <= 5:
                correct_5 += 1
            else:
                wrong_5 += 1
            if similarity <= 10:
                correct_10 += 1
            else:
                wrong_10 += 1
        else:
            if similarity > 1:
                correct_1 += 1
            else:
                wrong_1 += 1
            if similarity > 2:
                correct_2 += 1
            else:
                wrong_2 += 1
            if similarity > 5:
                correct_5 += 1
            else:
                wrong_5 += 1
            if similarity > 10:
                correct_10 += 1
            else:
                wrong_10 += 1

print("At similarity of 1: " + str(correct_1 / (correct_1 + wrong_1)))
print("At similarity of 2: " + str(correct_2 / (correct_2 + wrong_2)))
print("At similarity of 5: " + str(correct_5 / (correct_5 + wrong_5)))
print("At similarity of 10: " + str(correct_10 / (correct_10 + wrong_10)))

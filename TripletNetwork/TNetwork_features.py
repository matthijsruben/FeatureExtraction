import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers import Normalization
from keras.layers import LeakyReLU
import sys

data = pd.read_csv('../output/features/retain/features_categorised.csv').values.tolist()

train = [x for x in data if x[3] == "train"]
test = [x for x in data if x[3] == "test"]
validation_1 = [x for x in data if x[3] == "validation_1"]
validation_2 = [x for x in data if x[3] == "validation_2"]

y_train = [x[0] for x in train]
y_test = [x[0] for x in test]
y_validation_1 = [x[0] for x in validation_1]
y_validation_2 = [x[0] for x in validation_2]

classes = list(set([i[0] for i in data]))

x_train = np.array([x[4:11] + x[15:18] + x[22:] for x in train])
x_test = np.array([x[4:11] + x[15:18] + x[22:] for x in test])
x_validation_1 = np.array([x[4:11] + x[15:18] + x[22:] for x in validation_1])
x_validation_2 = np.array([x[4:11] + x[15:18] + x[22:] for x in validation_2])

X_train, Y_train = [], []
for label, v in zip(y_train, x_train):
    if Y_train.count(label) < 2:
        X_train.append(v)
        Y_train.append(label)
x_train = np.array(X_train)
y_train = np.array(Y_train)

input_neurons = x_train.shape[1]

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_validation_1 = x_validation_1.astype("float32")
x_validation_2 = x_validation_2.astype("float32")


def data_generator(batch_size=128):
    while True:
        a = []
        p = []
        n = []
        for _ in range(batch_size):
            pos_neg = random.sample(list(set(y_train)), 2)
            positive_sample = random.sample([d for d, l in zip(x_train, y_train) if l == pos_neg[0]], 2)
            negative_sample = random.choice([d for d, l in zip(x_train, y_train) if l == pos_neg[1]])
            a.append(positive_sample[0])
            p.append(positive_sample[1])
            n.append(negative_sample)
        yield [np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32")


def triplet_loss_l1(y_true, y_pred):
    anchor_out = y_pred[:, 0:output_neurons]
    positive_out = y_pred[:, output_neurons:(2 * output_neurons)]
    negative_out = y_pred[:, (2 * output_neurons):(3 * output_neurons)]

    pos_dist = K.sum(K.abs(anchor_out - positive_out), axis=1)
    neg_dist = K.sum(K.abs(anchor_out - negative_out), axis=1)

    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))


def triplet_loss_l2(y_true, y_pred):
    anchor_out = y_pred[:, 0:output_neurons]
    positive_out = y_pred[:, output_neurons:(2 * output_neurons)]
    negative_out = y_pred[:, (2 * output_neurons):(3 * output_neurons)]

    pos_dist = K.sqrt(K.sum(K.square(anchor_out - positive_out), axis=-1))
    neg_dist = K.sqrt(K.sum(K.square(anchor_out - negative_out), axis=-1))

    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))


def triplet_loss_euler(y_true, y_pred):
    anchor_out = y_pred[:, 0:output_neurons]
    positive_out = y_pred[:, output_neurons:(2 * output_neurons)]
    negative_out = y_pred[:, (2 * output_neurons):(3 * output_neurons)]

    pos_dist = K.sqrt(K.sum(K.square(anchor_out - positive_out), axis=-1))
    neg_dist = K.sqrt(K.sum(K.square(anchor_out - negative_out), axis=-1))

    delta_plus = K.exp(pos_dist) / (K.exp(pos_dist) + K.exp(neg_dist))
    delta_min = K.exp(neg_dist) / (K.exp(pos_dist) + K.exp(neg_dist))

    # return K.sqrt(K.sum(K.square(delta_plus - (delta_min - 1)), axis=-1)) ** 2
    return (delta_plus - (delta_min - 1)) ** 2

def triplet_loss_euler_2(y_true, y_pred):
    anchor_out = y_pred[:, 0:output_neurons]
    positive_out = y_pred[:, output_neurons:(2 * output_neurons)]
    negative_out = y_pred[:, (2 * output_neurons):(3 * output_neurons)]

    pos_dist = K.sqrt(K.sum(K.square(anchor_out - positive_out), axis=-1))
    neg_dist = K.sqrt(K.sum(K.square(anchor_out - negative_out), axis=-1))
    neg_dist_2 = K.sqrt(K.sum(K.square(positive_out - negative_out), axis=-1))
    neg_dist = tf.math.minimum(neg_dist_2, neg_dist)

    delta_plus = K.exp(pos_dist) / (K.exp(pos_dist) + K.exp(neg_dist))
    delta_min = K.exp(neg_dist) / (K.exp(pos_dist) + K.exp(neg_dist))

    return delta_plus ** 2 + ((1 - delta_min) ** 2)


def accuracies(thresholds, results, pprint=False):
    TP, TN = [0] * len(thresholds), [0] * len(thresholds)
    FP, FN = [0] * len(thresholds), [0] * len(thresholds)

    for i in range(len(results)):
        j = (i + 1) / len(results)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.flush()

        for j in range(len(results)):
            similarity = K.sqrt(K.sum(K.square(model_embeddings[i] - model_embeddings[j]), axis=-1))
            if y_validation_2[i] == y_validation_2[j]:
                for k in range(len(thresholds)):
                    if similarity <= thresholds[k]:
                        TP[k] += 1
                    else:
                        FN[k] += 1
            else:
                for k in range(len(thresholds)):
                    if similarity > thresholds[k]:
                        TN[k] += 1
                    else:
                        FP[k] += 1
    print("")
    if pprint:
        for i in range(len(thresholds)):
            print("At similarity of {}:".format(thresholds[i]))
            print("_________| Positive | Negative |\n"
                  "Positive | {:<7} | {:<7} |\n"
                  "Negative | {:<7} | {:<7} |".format(TP[i], FN[i], FP[i], TN[i]))
            recall = TP[i] / (TP[i] + FN[i])
            precision = TP[i] / (TP[i] + FP[i])
            true_negative = TN[i] / (TN[i] + FP[i])
            accuracy = (TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])
            print("Recall: {}, Precision: {}, Accuracy: {}".format(recall, precision, accuracy))
            print("Balanced accuracy : {}".format((recall + true_negative) / 2))
            print("")
    else:
        for i in range(len(thresholds)):
            print("At similarity of {}: {}".format(thresholds[i], str((TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i]))))


normalize = Normalization()#axis=1)
normalize.adapt(x_train)

# input_layer = Input(shape=(x_train.shape[1],))
# # x = normalize(input_layer)
# x = Dense(input_neurons, activation='relu')(input_layer)
# x = Dense(input_neurons, activation='relu')(x)
# x = Dense(output_neurons, activation='relu')(x)
# # model = Model(layer, x)
# model = Model(input_layer, x)

# model.summary()

output_neurons = x_train.shape[1]

# ReLu is best activation as: ReLu is fast
# https://stats.stackexchange.com/questions/218752/relu-vs-sigmoid-vs-softmax-as-hidden-layer-neurons
# https://datascience.stackexchange.com/questions/39042/how-to-use-leakyrelu-as-activation-function-in-sequence-dnn-in-keraswhen-it-per

model = Sequential([
    normalize,
    Dense(input_neurons, activation="relu"),
    # LeakyReLU(alpha=0.05),
    Dense(15, activation="relu"),
    # LeakyReLU(alpha=0.05),
    Dense(output_neurons, activation="relu")
    # LeakyReLU(alpha=0.05)
])

triplet_model_a = Input((input_neurons,))
triplet_model_p = Input((input_neurons,))
triplet_model_n = Input((input_neurons,))
triplet_model_out = Concatenate()([model(triplet_model_a), model(triplet_model_p), model(triplet_model_n)])
triplet_model = Model([triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)
# triplet_model.summary()

triplet_model.compile(loss=triplet_loss_euler, optimizer="adam")
triplet_model.fit(data_generator(), steps_per_epoch=200, epochs=15)

model_embeddings = triplet_model.layers[3].predict(x_validation_2, verbose=1)
print(model_embeddings.shape)

# reduced_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="correlation").fit_transform(model_embeddings)
# print(reduced_embeddings.shape)

# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=y_test, cmap="hsv")
# plt.show()
# import matplotlib.cm as cm
# colors = cm.rainbow(np.linspace(0, 1, reduced_embeddings.shape[0]))
# for dataset, color in zip(reduced_embeddings, colors):
#     plt.scatter(y_test, dataset, color=color)
#
#
# # for i, txt in enumerate(y_test):
# #     plt.annotate(txt, (reduced_embeddings[:, 0][i], reduced_embeddings[:, 1][i]))
# plt.show()


accuracies([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], model_embeddings, True)

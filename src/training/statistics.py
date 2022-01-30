import concurrent.futures
import os
import sys

import pandas as pd
from tensorflow.keras import backend as K


def calc_euclid_distance(a, b):
    return K.sqrt(K.sum(K.square(a - b), axis=-1))


def calc_manhattan_distance(a, b):
    return K.sum(K.abs(a - b), axis=-1, keepdims=True)


def generate_pairs(model_outputs, y):
    output_pairs = []
    y_pairs = []
    for i in range(len(model_outputs)):
        for j in range(min(i + 1, len(model_outputs)), len(model_outputs)):
            output_pairs.append((model_outputs[i], model_outputs[j]))
            y_pairs.append((y.iloc[i], y.iloc[j]))

    return output_pairs, y_pairs


def get_similarity_label_pairs(output_pairs, y_pairs, similarity_measure, thresholds):
    tp, tn = [0] * len(thresholds), [0] * len(thresholds)
    fp, fn = [0] * len(thresholds), [0] * len(thresholds)

    similarity_label_pairs = []
    for i in range(len(output_pairs)):
        pair = output_pairs[i]
        y_pair = y_pairs[i]
        similarity_label_pairs.append((similarity_measure(pair[0], pair[1]), y_pair[0] == y_pair[1]))

    for similarity, same_label in similarity_label_pairs:
        for i, threshold in enumerate(thresholds):
            if same_label:
                if similarity <= threshold:
                    tp[i] += 1
                else:
                    fn[i] += 1
            else:
                if similarity > threshold:
                    tn[i] += 1
                else:
                    fp[i] += 1

    return tp, tn, fp, fn


def generate_confusion_matrix(thresholds, model_outputs, y, similarity_measure):
    """
    Returns the confusion matrix related to the model performance based on the outputs of the model and a similarity
    threshold, the latter of which decides whether two inputs are similar or dissimilar.
    :param thresholds: List of floats of similarity thresholds to check.
    :param model_outputs: List of outputs of the embedded model
    :param y: The true label for each of the outputs of the embedded model
    :param similarity_measure: The measure to use for calculating the similarity between two outputs
    :return: ([true positives for each threshold], [true negatives for each threshold], [false positives for each
    threshold], [false negatives for each threshold])
    """
    output_pairs, y_pairs = generate_pairs(model_outputs, y)
    batch_size = 1000

    tp, tn = [0] * len(thresholds), [0] * len(thresholds)
    fp, fn = [0] * len(thresholds), [0] * len(thresholds)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, len(output_pairs), batch_size):
            output_pairs_batch = output_pairs[i:min(len(output_pairs), i + batch_size)]
            y_pairs_batch = y_pairs[i:min(len(y_pairs), i + batch_size)]

            futures.append(executor.submit(get_similarity_label_pairs, output_pairs_batch, y_pairs_batch,
                                           similarity_measure, thresholds))

        progress = 0
        for future in concurrent.futures.as_completed(futures):
            progress += 1
            batch_tp, batch_tn, batch_fp, batch_fn = future.result()

            for i in range(len(thresholds)):
                tp[i] += batch_tp[i]
                tn[i] += batch_tn[i]
                fp[i] += batch_fp[i]
                fn[i] += batch_fn[i]

            j = (progress + 1) / (len(output_pairs) // batch_size)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
            sys.stdout.flush()

    # Ensure subsequent lines are printed on a  new line
    print("")

    return tp, tn, fp, fn


def calc_recall(tpos, fneg):
    if tpos == 0 and fneg == 0:
        return 0

    return tpos / (tpos + fneg)


def calc_precision(tpos, fpos):
    if tpos == 0 and fpos == 0:
        return 0

    return tpos / (tpos + fpos)


def calc_accuracy(tpos, tneg, fpos, fneg):
    return (tpos + tneg) / (tpos + tneg + fpos + fneg)


def calc_balanced_accuracy(tpos, fpos, tneg, fneg):
    tneg_rate = tneg / (tneg + fpos)
    return (calc_recall(tpos, fneg) + tneg_rate) / 2


def calc_f1_score(tpos, fpos, fneg):
    precision = calc_precision(tpos, fpos)
    recall = calc_recall(tpos, fneg)

    if precision == 0 and recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)


def generate_statistics(thresholds, model_outputs, y, similarity_measure=None, pprint=False):
    """
    Determines the statistics related to the model performance based on the outputs of the model and a similarity
    threshold, the latter of which decides whether two inputs are similar or dissimilar.
    :param thresholds: List of floats of similarity thresholds to check.
    :param model_outputs: List of outputs of the embedded model
    :param y: The true label for each of the outputs of the embedded model
    :param similarity_measure: The function to use for calculating similarity
    :param pprint: Whether to pretty-print the statistics
    :return: A list of tuples in the format (threshold, [statistic: statistic-value])
    """
    tp, tn, fp, fn = generate_confusion_matrix(thresholds, model_outputs, y, similarity_measure=calc_manhattan_distance)

    statistics = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]

        recall = calc_recall(tp[i], fn[i])
        precision = calc_precision(tp[i], fp[i])
        accuracy = calc_accuracy(tp[i], tn[i], fp[i], fn[i])
        balanced_accuracy = calc_balanced_accuracy(tp[i], fp[i], tn[i], fn[i])
        f1_score = calc_f1_score(tp[i], fp[i], fn[i])

        statistics.append((
            threshold, {
                "recall": recall,
                "precision": precision,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_score": f1_score,
            }))

        if pprint:
            print("At similarity of {}:".format(thresholds[i]))
            print("_________| Positive | Negative |\n"
                  "Positive | {:<7} | {:<7} |\n"
                  "Negative | {:<7} | {:<7} |".format(tp[i], fn[i], fp[i], tn[i]))
            print("Recall: {}, Precision: {}, Accuracy: {}".format(recall, precision, accuracy))
            print("Balanced accuracy : {}".format(balanced_accuracy))
            print("F1-score: {}".format(f1_score))
            print("")

    return statistics


def write_statistics(statistics, filename):
    dicts = []
    for threshold, stats in statistics:
        dicts.append({"threshold": threshold, **stats})

    df = pd.DataFrame(dicts)
    os.makedirs("./output/network-stats", exist_ok=True)
    df.to_csv(f"./output/network-stats/{filename}.csv", index=False)

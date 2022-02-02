import sys

from tensorflow.keras import backend as K


def generate_confusion_matrix(thresholds, model_outputs, y):
    """
    Returns the confusion matrix related to the model performance based on the outputs of the model and a similarity
    threshold, the latter of which decides whether two inputs are similar or dissimilar.
    :param thresholds: List of floats of similarity thresholds to check.
    :param model_outputs: List of outputs of the embedded model
    :param y: The true label for each of the outputs of the embedded model
    :return: ([true positives for each threshold], [true negatives for each threshold], [false positives for each
    threshold], [false negatives for each threshold])
    """
    tp, tn = [0] * len(thresholds), [0] * len(thresholds)
    fp, fn = [0] * len(thresholds), [0] * len(thresholds)

    print(len(model_outputs))
    for i in range(len(model_outputs)):
        j = (i + 1) / len(model_outputs)
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.flush()

        for j in range(len(model_outputs)):
            similarity = K.sqrt(K.sum(K.square(model_outputs[i] - model_outputs[j]), axis=-1))

            if y.iloc[i] == y.iloc[j]:
                for k in range(len(thresholds)):
                    if similarity <= thresholds[k]:
                        tp[k] += 1
                    else:
                        fp[k] += 1
            else:
                for k in range(len(thresholds)):
                    if similarity > thresholds[k]:
                        tn[k] += 1
                    else:
                        fn[k] += 1

    return tp, tn, fp, fn


def calc_recall(tpos, fneg):
    return tpos / (tpos + fneg)


def calc_precision(tpos, fpos):
    return tpos / (tpos + fpos)


def calc_accuracy(tpos, tneg, fpos, fneg):
    return (tpos + tneg) / (tpos + tneg + fpos + fneg)


def generate_statistics(thresholds, model_outputs, y, pprint=False):
    """
    Determines the statistics related to the model performance based on the outputs of the model and a similarity
    threshold, the latter of which decides whether two inputs are similar or dissimilar.
    :param thresholds: List of floats of similarity thresholds to check.
    :param model_outputs: List of outputs of the embedded model
    :param y: The true label for each of the outputs of the embedded model
    :param pprint: Whether to pretty-print the statistics
    :return: A list of tuples in the format (threshold, [statistic: statistic-value])
    """
    tp, tn, fp, fn = generate_confusion_matrix(thresholds, model_outputs, y)

    statistics = []
    for i in range(len(thresholds)):
        threshold = thresholds[i]

        recall = calc_recall(tp[i], fn[i])
        precision = calc_precision(tp[i], fp[i])
        accuracy = calc_accuracy(tp[i], tn[i], fp[i], fn[i])

        statistics.append((
            threshold, {
                "recall": recall,
                "precision": precision,
                "accuracy": accuracy,
            }))

        if pprint:
            print("At similarity of {}:".format(thresholds[i]))
            print("_________| Positive | Negative |\n"
                  "Positive | {:<7} | {:<7} |\n"
                  "Negative | {:<7} | {:<7} |".format(tp[i], fn[i], fp[i], tn[i]))
            print("Recall: {}, Precision: {}, Accuracy: {}".format(recall, precision, accuracy))
            print("Balanced accuracy : {}".format((recall + (1 - recall)) / 2))
            print("")

    return statistics

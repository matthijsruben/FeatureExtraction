import gzip
import numpy as np


# loads training or validation/testing data depending on the value of train_or_val
def load_data(train_or_val):
    if train_or_val:
        # open files for TRAINING DATA
        f_images = gzip.open('MNIST_data/train-images-idx3-ubyte.gz', 'rb')
        f_labels = gzip.open('MNIST_data/train-labels-idx1-ubyte.gz', 'rb')
    else:
        # open files for VALIDATION/TEST DATA
        f_images = gzip.open('MNIST_data/t10k-images-idx3-ubyte.gz', 'rb')
        f_labels = gzip.open('MNIST_data/t10k-labels-idx1-ubyte.gz', 'rb')

    # convert bytes to array of type uint8
    buf_images = f_images.read()
    buf_labels = f_labels.read()
    f_images.close()
    f_labels.close()
    images = np.frombuffer(buf_images, dtype=np.uint8).astype(float)
    labels = np.frombuffer(buf_labels, dtype=np.uint8)

    # First 16, 8 (resp.) are metadata and not pixels, labels (resp.)
    images = images[16:]
    labels = labels[8:]

    # convert array in pixel values to array of vectors (of 784 pixels)
    images = images.reshape((int(len(images)/784), 784, 1))
    # convert array of labels (number 0-10) to array of corresponding label vectors (e.g. [0 0 1 0 0 0 0 0 0 0] for 3)
    labels = [create_vector(i) for i in labels]

    # convert gray scale value to values between 0 and 1
    images = np.divide(images, 255.0)

    return images, labels


def create_vector(i):
    vector = np.zeros((10, 1))
    vector[i] = 1.0
    return vector
import numpy as np
import tarfile
import cv2
import matplotlib.pyplot as plt
import random


# function to plot image using matplotlib
def show(pixels):
    plt.imshow(pixels, cmap='gray')
    plt.show()


# a function that, provided a tarfile, returns a dict data with:
# key: a writer (string)
# val: a list of images (array)
# each image is a 2-dimensional array with grayscale pixel-values (0-255)
def load_data(path):
    data = {}
    with tarfile.open(path, 'r:gz') as tar:
        for member, name in zip(tar.getmembers(), tar.getnames()):
            f = tar.extractfile(member)
            if f is not None:
                content = f.read()
                f.close()
                """
                    Useful flags for decoding:
                    -1  :   Return the loaded image as is (unchanged)
                    0   :   Convert image to single channel grayscale image
                    16  :   Convert image to single channel grayscale image and image size reduced 1/2
                    32  :   Convert image to single channel grayscale image and image size reduced 1/4
                    64  :   Convert image to single channel grayscale image and image size reduced 1/8
                """
                image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), -1)

                # add key and value to data dict
                writer = name.split('/')[1]
                if writer not in data.keys():
                    data[writer] = []
                data[writer].append(image)

    return data


# Load all data into a dictionary
data = load_data('data/lines.tgz')

# Choose a writer randomly and consequently choose one of their lines randomly and show it
sample = random.sample(data.items(), 1)
sample_writer = sample[0][0]
print(sample_writer)
show(random.sample(sample[0][1], 1)[0])

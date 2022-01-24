import MNISTloader
import TripletNetwork
import matplotlib.pyplot as plt
import numpy as np

images, labels = MNISTloader.load_data(True)
training_data = [(images[i], labels[i]) for i in range(int(len(images)))]
net = TripletNetwork.Network([784, 100, 1], 0)
net.SGD(training_data, 25, 10, 0.05)


# def filter_classes(example, until):
#     c = list(example[1]).index(1)
#     return c in range(until)
#
#
# plot_data = []
# for split in range(1, 10):
#     splitted_data = list(filter(lambda example: filter_classes(example, split + 1), training_data))
#     net = TripletNetwork.Network([784, 100, 1], 0)
#     epochs_axis, losses, accuracies = net.SGD(splitted_data, 5, 10, 0.05, split + 1)
#     plot_data.append((epochs_axis, losses, accuracies))
#
# # Plot accuracies
# for info in plot_data:
#     plt.plot(info[0], info[2])
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(["2 classes", "3 classes", "4 classes", "5 classes", "6 classes",
#             "7 classes", "8 classes", "9 classes", "10 classes"], prop={"size": 5})
# plt.show()
#
# # Plot  losses
# for info in plot_data:
#     plt.plot(info[0], info[1])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(["2 classes", "3 classes", "4 classes", "5 classes", "6 classes",
#             "7 classes", "8 classes", "9 classes", "10 classes"], prop={"size": 5})
# plt.show()

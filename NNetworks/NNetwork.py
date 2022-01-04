import numpy as np
import random


class Nnetwork:
    num_layers = 0
    sizes = None
    biases = None
    weights = None

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative Activation function
    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b[0])
        return a

    def SGD(self, training_data, batch_size, epochs, eta):
        for x in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.update_batch(batch, eta)

    def update_batch(self, batch, eta):
        bias_change = [np.zeros(b.shape) for b in self.biases]
        weight_change = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in batch:
            delta_b, delta_w = self.backprop(x, y)
            bias_change = [cb + db for cb, db in zip(bias_change, delta_b)]
            weight_change = [cw + dw for cw, dw in zip(weight_change, delta_w)]
        self.biases = [old - change * (eta / len(batch)) for old, change in zip(self.biases, bias_change)]
        self.weights = [old - change * (eta / len(batch)) for old, change in zip(self.weights, weight_change)]

    def backprop(self, x, y):
        bias_change = [np.zeros(b.shape) for b in self.biases]
        weight_change = [np.zeros(w.shape) for w in self.weights]

        # feed forward
        a = x
        activations = [x]  # list to store activations
        zs = []  # list of z-vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, a) + b[0]
            zs.append(a)
            a = self.sigmoid(a)
            activations.append(a)

        # back propagate
        delta = self.loss_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        bias_change[-1] = delta
        weight_change[-1] = np.outer(delta, activations[-2])

        for l in range(2, self.num_layers):
            sp = self.sigmoid_prime(zs[-l])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            bias_change[-l] = delta
            weight_change[-l] = np.outer(delta, activations[-l - 1])
        return bias_change, weight_change

    def loss_derivative(self, x, y):
        return 2 * (x-y)


input_1 = [0.2, 0.3, 0.8]
input_2 = [0.6, 0.1, 0.4]
neural = Nnetwork([3, 4, 3])

print("input 1: ")
print(neural.feed_forward(input_1))
print("input 2: ")
print(neural.feed_forward(input_2))

poss = [([0.2, 0.3, 0.8], [0.7, 0.1, 0.7])] * 5000
negg = [([0.6, 0.1, 0.4], [0.9, 0.8, 0.1])] * 5000

train = poss
train.extend(negg)

neural.SGD(train, 2, 3, 1)
#[0.7, 0.1, 0.7]
print("input 1 after training: ")
print(neural.feed_forward(input_1))
#[0.9, 0.8, 0.1]
print("input 2 after training: ")
print(neural.feed_forward(input_2))


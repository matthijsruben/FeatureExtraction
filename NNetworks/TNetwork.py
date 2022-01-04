import numpy as np
import random
import itertools as it


class Tnetwork:
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

    # Derivative activation function
    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b[0])
            # a = np.dot(w, a) + b[0]
        return a

    def SGD(self, training_data, batch_size, epochs, eta):
        for i in range(epochs):
            labeled = []
            for x in training_data:
                anchor = x.copy()
                random.shuffle(x)
                pos_labels = list(zip(x, anchor))
                all_labels = []
                for y in training_data:
                    if y is not x:
                        random.shuffle(y)  # TODO check if shuffle is needed
                        all_labels = [a + (b,) for a, b in zip(pos_labels, it.cycle(y))] \
                            if len(y) < len(pos_labels) \
                            else [a + (b,) for a, b in zip(pos_labels, y)]
                labeled.extend(all_labels)
            batches = [labeled[k:k + batch_size] for k in range(0, len(labeled), batch_size)]
            random.shuffle(batches)
            for batch in batches:
                self.update_batch(batch, eta)

    def update_batch(self, batch, eta):
        bias_change = [np.zeros(b.shape) for b in self.biases]
        weight_change = [np.zeros(w.shape) for w in self.weights]
        for (anchor, pos, neg) in batch:
            delta_b, delta_w = self.backprop(anchor, pos, neg)
            bias_change = [cb + db for cb, db in zip(bias_change, delta_b)]
            weight_change = [cw + dw for cw, dw in zip(weight_change, delta_w)]
        self.biases = [old - (change * eta / len(batch)) for old, change in zip(self.biases, bias_change)]
        self.weights = [old - (change * eta / len(batch)) for old, change in zip(self.weights, weight_change)]

    def backprop(self, anchor, pos, neg):
        bias_change = [np.zeros(b.shape) for b in self.biases]
        weight_change = [np.zeros(w.shape) for w in self.weights]

        # feed forward
        a = anchor
        activations = [anchor]  # list to store activations
        zs = []  # list of z-vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, a) + b[0]
            zs.append(a)
            a = self.sigmoid(a)
            activations.append(a)

        pos_values = self.feed_forward(pos)
        neg_values = self.feed_forward(neg)

        # back propagate
        delta = self.loss_derivative(activations[-1], pos_values, neg_values) * self.sigmoid_prime(zs[-1])

        bias_change[-1] = delta
        weight_change[-1] = np.outer(delta, activations[-2])

        for l in range(2, self.num_layers):
            sp = self.sigmoid_prime(zs[-l])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            bias_change[-l] = delta
            weight_change[-l] = np.outer(delta, activations[-l - 1])
        return bias_change, weight_change

    def loss_derivative(self, x, y, z):
        delta_plus = np.exp(np.linalg.norm(x-y)) / (np.exp(np.linalg.norm(x-y)) + np.exp(np.linalg.norm(x-z)))
        delta_min = np.exp(np.linalg.norm(x-z)) / (np.exp(np.linalg.norm(x-y)) + np.exp(np.linalg.norm(x-z)))
        y_deriv = 2 * delta_plus * (delta_plus * (delta_plus - 1) + delta_min * (delta_min - 1))
        z_deriv = 2 * delta_min * (delta_plus ** 2 + (delta_min - 1) ** 2)
        return z_deriv

input_1 = [0.2, 0.3, 0.3]
input_2 = [0.6, 0.7, 0.9]
neural = Tnetwork([3, 3, 3])

print("input 1:")
print(neural.feed_forward(input_1))
print("input 2:")
print(neural.feed_forward(input_2))

high = []
low = []
i = 0.1
while i < 0.5:
    j = 0.1
    while j < 0.5:
        k = 0.1
        while k < 0.5:
            high.append([i, j, k])
            low.append([i+0.4, j+0.4, k+0.4])
            k = k + 0.1
        j = j + 0.1
    i = i + 0.1


train = [high] + [low]

neural.SGD(train, 4, 100, 1)

print("input 1 after training:")
print(neural.feed_forward(input_1))
print("input 2 after training:")
print(neural.feed_forward(input_2))

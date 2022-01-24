import random
import numpy as np
import matplotlib.pyplot as plt
import functools


class Network(object):

    def __init__(self, sizes, seed):
        """"A network can be initialised with an array called sizes. For example if sizes was [2, 3, 1], then it would
            be a 3-layer network, with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer
            1 neuron.
            Currently biases and weights are randomly generated from a standard normal distribution with mean 0 and variance 1.
            The first layer is the input layer, so it has no biases.
            Seed is for generating the same random numbers every run. Useful for comparing methods."""
        self.amountLayers = len(sizes)
        self.sizes = sizes
        # Initialize each weight using a Gaussian distribution with mean = "zero" and standard deviation =
        # "one over the square root of the number of weights connecting to the same neuron". (REGULARIZATION TECHNIQUE)
        # So not the standard normal distribution (mu = 0, sigma = 1)
        np.random.seed(seed)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        np.random.seed(seed+1)
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward_output(self, a):
        """"Returns the output vector of the network, if the vector a is the input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def feedforward(self, a):
        """"Returns two arrays. The first is the list of all activation_vectors ordered by layer.
        The second is the list of all weighted-input-sum_vectors ordered by layer"""
        activation = a
        activations = [a]
        weighted_input_sums = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            weighted_input_sums.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations, weighted_input_sums

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate):  # use extra parameter split to compare amount of classes
        """"Performs stochastic gradient descent (SGD) on the network, using mini batches. Training data is a list of
        tuples (x,y) representing the training inputs and the desired outputs. The training data is shuffled and
        divided into mini_batches. For each mini_batch gradient descent is performed on all weights and biases in the
        network. Gradient descent will be performed for all mini batch each epoch. Also, metrics are computed, printed,
        and shown in a plot"""
        training_data = list(training_data)
        splits = [list(filter(lambda example: (example[1] == create_vector(i)).all(), training_data)) for i in range(10)]  # use split instead of 10

        # Initial prints, start showing initial metrics before training
        initial_loss, initial_accuracy = self.calculate_metrics(training_data, splits)
        print_initial_metrics("Stochastic Gradient Descent (SGD)", initial_loss, initial_accuracy)
        losses = [initial_loss]
        accuracies = [initial_accuracy]
        epochs_axis = [0]

        # Repeat the process every epoch
        for i in range(epochs):
            # shuffle the training_data and divide into mini_batches
            random.seed(0)
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]

            # calculate the average batch gradient for weights and biases and use it to perform GRADIENT DESCENT
            for mini_batch in mini_batches:
                batch_gradient_bias, batch_gradient_weights = self.calculate_batch_gradient(mini_batch, splits)
                self.gradient_descent(batch_gradient_bias, batch_gradient_weights, learning_rate)

            # METRICS calculation
            loss, accuracy = self.calculate_metrics(training_data, splits)
            losses.append(loss)
            accuracies.append(accuracy)
            epochs_axis.append(i + 1)

            # PRINT in console
            print_training_metrics(i + 1, loss, accuracy)

        # PLOT the metrics
        plot_metrics(epochs_axis, losses, accuracies)
        # return epochs_axis, losses, accuracies  Uncomment when comparing different amount of classes

    def calculate_metrics(self, training_data, splits):
        """"Returns the metrics loss and accuracy for all training_data after an epoch"""
        correct_predictions = 0
        los_sum = 0
        for training_example in range(len(training_data)):
            # Find a corresponding triplet
            anchor, pos, neg = self.find_triplet(training_data[training_example], splits)
            output_anchor, output_pos, output_neg = self.feedforward_output(anchor[0]), self.feedforward_output(pos[0]), self.feedforward_output(neg[0])

            if self.predict(output_anchor, output_pos, output_neg):
                correct_predictions += 1
            los_sum += self.triplet_loss(output_anchor, output_pos, output_neg)

        loss = los_sum / len(training_data)
        accuracy = correct_predictions / len(training_data)

        return loss, accuracy

    def predict(self, output_anchor, output_positive, output_negative):
        """"Returns True if the distance (L2-norm) between the output_anchor & output_negative is larger than the
            distance (L2-norm) between the output_anchor & output_positive. In other words, returns True if the
            prediction is correct, else False."""
        return np.linalg.norm(output_anchor - output_negative) > np.linalg.norm(output_anchor - output_positive)

    def find_triplet(self, anchor, splits):
        """"Based on training_data that is split into classes: splits = [ [(x,0), (x,0),...], [(x,1), (x,1),...], ... ]
            and based on the anchor example, return a triplet (anchor, positive, negative), e.g. ((x,5), (x,5), (x,8)),
            where the positive example is a randomly chosen example of the same class as the anchor and
            the negative example is a randomly chosen example of a randomly chosen different class from the anchor"""
        # one-hot-encoded vector back into integer
        anchor_class = list(anchor[1]).index(1)
        different_class = random.choice(list(filter(lambda i: i is not anchor_class, [i for i in range(len(splits))])))

        positive_example = random.choice(splits[anchor_class])
        negative_example = random.choice(splits[different_class])

        return anchor, positive_example, negative_example

    def calculate_batch_gradient(self, mini_batch, splits):
        """"Returns the average gradient over the batch of all the biases and all the weights in the network.
        Mini_batch is a list of tuples (x,y) representing a batch of training inputs and desired outputs."""
        # GRADIENT biases initialized as empty list of pdb vectors and GRADIENT weights empty list of pdw vectors
        batch_gradient_bias = [[np.zeros(b.shape) for b in self.biases] for i in range(3)]
        batch_gradient_weights = [[np.zeros(w.shape) for w in self.weights] for i in range(3)]

        # for each training example in the mini-batch, calculate update for all weights and biases in the network
        # the gradients of all training examples in the mini batch will be added up in the batch_gradient
        for i in range(len(mini_batch)):
            # CREATE TRIPLET
            anchor, pos, neg = self.find_triplet(mini_batch[i], splits)
            triplet = [anchor, pos, neg]

            # Calculate derivative of the loss with respect to anchor, positive, and negative
            output_anchor = self.feedforward_output(anchor[0])
            output_pos = self.feedforward_output(pos[0])
            output_neg = self.feedforward_output(neg[0])
            pd_anchor, pd_pos, pd_neg = self.triplet_loss_derivative(output_anchor, output_pos, output_neg)
            triplet_loss_derivative = [pd_anchor, pd_pos, pd_neg]

            for j in range(len(triplet)):
                # FEEDFORWARD
                activations, weighted_input_sums = self.feedforward(triplet[j][0])

                # pdbv = (per-layer) vector of partial derivatives of the loss function with respect to the bias
                # pdwv = (per-layer) matrix of partial derivatives of the loss function with respect to the weight
                pdbv = sigmoid_derivative(weighted_input_sums[-1]) * triplet_loss_derivative[j]
                pdwv = np.dot(pdbv, np.transpose(activations[-2]))

                # pdbv and pdwv that were just initialized are now added to the list of partial derivatives
                # this list is called the gradient
                gradient_bias = [pdbv]
                gradient_weights = [pdwv]

                # BACKPROPAGATION
                # start from 2, because the pdbv and pdwv of the last layer are already calculated and added.
                for k in range(2, self.amountLayers):
                    pdbv = sigmoid_derivative(weighted_input_sums[-k]) * np.dot(self.weights[-k+1].transpose(), pdbv)
                    pdwv = np.dot(pdbv, np.transpose(activations[-k-1]))
                    gradient_bias.append(pdbv)
                    gradient_weights.append(pdwv)

                # pdb/pdw vectors are added in order from last layer to first layer. Reverse for later purposes
                gradient_bias.reverse()
                gradient_weights.reverse()

                # Add each pdbv and pdwv of the gradient to the corresponding vector of the batch_gradient
                for pdb_vector in range(len(gradient_bias)):
                    batch_gradient_bias[j][pdb_vector] += gradient_bias[pdb_vector]
                for pdw_vector in range(len(gradient_weights)):
                    batch_gradient_weights[j][pdw_vector] += gradient_weights[pdw_vector]

        # Finally, the sum that was added up in the batch_gradient is divided by the amount of training examples from
        # the mini-batch to get the average gradient over the mini-batch
        for j in range(len(triplet)):
            batch_gradient_bias[j] = [pdb / len(mini_batch) for pdb in batch_gradient_bias[j]]
            batch_gradient_weights[j] = [pdw / len(mini_batch) for pdw in batch_gradient_weights[j]]

        # Take the average gradient over the 3 elements of the triple
        batch_gradient_bias = functools.reduce(lambda a, b: [np.add(el1, el2) for el1, el2 in zip(a, b)], batch_gradient_bias)
        batch_gradient_bias = [pdb_sum / 3 for pdb_sum in batch_gradient_bias]
        batch_gradient_weights = functools.reduce(lambda a, b: [np.add(el1, el2) for el1, el2 in zip(a, b)], batch_gradient_weights)
        batch_gradient_weights = [pdw_sum / 3 for pdw_sum in batch_gradient_weights]

        return batch_gradient_bias, batch_gradient_weights

    def gradient_descent(self, gradient_bias, gradient_weights, learning_rate):
        """"Updates all the weights and biases in the network by taking a step in the negative gradient direction
        of the loss function with respect to each particular weight and bias in the network. The step is multiplied
        by the learning rate in order to control the step size"""
        # STEP SIZE calculation
        update_step_biases = [learning_rate * (-1) * pdbv for pdbv in gradient_bias]
        update_step_weights = [learning_rate * (-1) * pdwv for pdwv in gradient_weights]

        # update all weights and biases in the network, done by (per-layer) vector-wise addition
        for elem in range(self.amountLayers - 1):
            self.biases[elem] += update_step_biases[elem]
            self.weights[elem] += update_step_weights[elem]

    def triplet_loss(self, output_anchor, output_positive, output_negative):
        exp_pos_diff = np.exp(np.linalg.norm(output_anchor - output_positive))
        exp_neg_diff = np.exp(np.linalg.norm(output_anchor - output_negative))

        # Softmax: d_plus = 1 - d_minus
        d_plus = exp_pos_diff / (exp_pos_diff + exp_neg_diff)
        d_minus = exp_neg_diff / (exp_pos_diff + exp_neg_diff)

        return np.linalg.norm(np.array([d_plus, d_minus]) - np.array([0, 1])) ** 2

    def triplet_loss_derivative(self, output_anchor, output_positive, output_negative):
        exp_pos_diff = np.exp(np.linalg.norm(output_anchor - output_positive))
        exp_neg_diff = np.exp(np.linalg.norm(output_anchor - output_negative))

        # Softmax: d_plus = 1 - d_minus
        d_plus = exp_pos_diff / (exp_pos_diff + exp_neg_diff)
        d_minus = exp_neg_diff / (exp_pos_diff + exp_neg_diff)

        # Note that because of softmax effects, pd_positive = - pd_negative
        pd_anchor = 0
        pd_positive = 2 * d_plus * (d_plus * (d_plus - 1) + d_minus * (d_minus - 1))
        pd_negative = 2 * d_minus * (d_plus ** 2 + (d_minus - 1) ** 2)

        return pd_anchor, pd_positive, pd_negative


# Some math functions
def sigmoid(z):
    """"Returns the sigmoid function"""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    """"Returns the derivative of the sigmoid function"""
    return sigmoid(z)*(1-sigmoid(z))


# Some printing and plotting functions
def print_initial_metrics(method, initial_loss, initial_accuracy):
    print("Start Training: 0 epochs complete")
    print("Training method: " + method)
    print("Initial loss: {}".format(initial_loss))
    print("Initial accuracy: {} \n".format(initial_accuracy))


def print_training_metrics(epoch, loss, accuracy):
    print("Epoch {} complete".format(epoch))
    print("Loss: {}".format(loss))
    print("Accuracy: {} \n".format(accuracy))


def plot_metrics(horizontal_axis, losses, accuracies):
    plt.plot(horizontal_axis, losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    plt.plot(horizontal_axis, accuracies)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


def create_vector(i):
    vector = np.zeros((10, 1))
    vector[i] = 1.0
    return vector


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

class SimpleDNN:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
            b = np.random.randn(layer_sizes[i+1], 1) * 0.1
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        activations = [x]
        pre_activations = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            pre_activations.append(z)
            a = sigmoid(z)
            activations.append(a)
        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true):
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        y_pred = activations[-1]
        delta = mse_loss_grad(y_pred, y_true) * sigmoid_deriv(pre_activations[-1])
        grads_w[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = delta
        for l in range(self.num_layers - 2, -1, -1):
            delta = np.dot(self.weights[l+1].T, delta) * sigmoid_deriv(pre_activations[l])
            grads_w[l] = np.dot(delta, activations[l].T)
            grads_b[l] = delta
        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b):
        for l in range(self.num_layers):
            self.weights[l] -= self.learning_rate * grads_w[l]
            self.biases[l]  -= self.learning_rate * grads_b[l]

    def train_step(self, x, y_true):
        activations, pre_activations = self.forward(x)
        loss = mse_loss(activations[-1], y_true)
        grads_w, grads_b = self.backward(activations, pre_activations, y_true)
        self.update_parameters(grads_w, grads_b)
        return loss, activations[-1]

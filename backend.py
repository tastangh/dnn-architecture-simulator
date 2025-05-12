import numpy as np

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    return dA * s * (1 - s)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def calculate_loss(AL, Y, loss_type="mse"):
    m = Y.shape[1]
    epsilon = 1e-8

    if loss_type == "cross_entropy":
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        return -np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped)) / m
    elif loss_type == "mae":
        return np.mean(np.abs(AL - Y))
    elif loss_type == "mse":
        return np.mean((AL - Y) ** 2)
    elif loss_type == "rmse":
        return np.sqrt(np.mean((AL - Y) ** 2))
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")

def loss_gradient(AL, Y, loss_type="mse"):
    m = Y.shape[1]
    epsilon = 1e-8
    if loss_type == "cross_entropy":
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        return - (np.divide(Y, AL_clipped) - np.divide(1 - Y, 1 - AL_clipped)) / m
    elif loss_type == "mae":
        return np.sign(AL - Y) / m
    elif loss_type == "mse":
        return 2 * (AL - Y) / m
    elif loss_type == "rmse":
        mse = np.mean((AL - Y) ** 2)
        rmse = np.sqrt(mse + epsilon)
        return (AL - Y) / (m * rmse)
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")

class DeepDNN:
    def __init__(self, layer_sizes, learning_rate=0.01, hidden_activation="relu", loss_type="mse"):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.loss_type = loss_type
        self.num_layers = len(layer_sizes) - 1
        self.parameters = {}
        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(3)
        for l in range(1, len(self.layer_sizes)):
            self.parameters[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * 0.01
            self.parameters[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

    def _activation_forward(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        if activation == "sigmoid":
            A, cache = sigmoid(Z)
        elif activation == "relu":
            A, cache = relu(Z)
        else:
            raise ValueError("Unsupported activation")
        return A, (A_prev, W, b, cache)

    def forward(self, X):
        A = X
        caches = []
        for l in range(1, self.num_layers):
            A, cache = self._activation_forward(A, self.parameters[f"W{l}"], self.parameters[f"b{l}"], self.hidden_activation)
            caches.append(cache)
        AL, cache = self._activation_forward(A, self.parameters[f"W{self.num_layers}"], self.parameters[f"b{self.num_layers}"], "sigmoid")
        caches.append(cache)
        return AL, caches

    def _activation_backward(self, dA, cache, activation):
        A_prev, W, b, Z = cache
        if activation == "relu":
            dZ = relu_backward(dA, Z)
        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, Z)
        else:
            raise ValueError("Unsupported activation")
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def backward(self, AL, Y, caches):
        grads = {}
        L = self.num_layers
        dAL = loss_gradient(AL, Y, self.loss_type)
        current_cache = caches[-1]
        grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = self._activation_backward(dAL, current_cache, "sigmoid")
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev, dW, db = self._activation_backward(grads[f"dA{l+1}"], current_cache, self.hidden_activation)
            grads[f"dA{l}"] = dA_prev
            grads[f"dW{l+1}"] = dW
            grads[f"db{l+1}"] = db
        return grads

    def update_parameters(self, grads):
        for l in range(1, self.num_layers + 1):
            self.parameters[f"W{l}"] -= self.learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * grads[f"db{l}"]

    def train_step(self, X, Y):
        AL, caches = self.forward(X)
        loss = calculate_loss(AL, Y, self.loss_type)
        grads = self.backward(AL, Y, caches)
        self.update_parameters(grads)
        return loss, AL

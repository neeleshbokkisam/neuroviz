import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1

        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        self.activations = []
        self.z_values = []

    def _activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"unknown activation: {self.activation}")

    def _activate_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError(f"unknown activation: {self.activation}")

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        current = X

        for i in range(self.num_layers - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current = self._activate(z)
            self.activations.append(current)

        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        self.activations.append(output)

        return output

    def backward(self, X, y, output):
        m = X.shape[0]
        dW = [np.zeros_like(w) for w in self.weights]
        dB = [np.zeros_like(b) for b in self.biases]

        dz = output - y

        for i in range(self.num_layers - 1, -1, -1):
            dW[i] = (1 / m) * np.dot(self.activations[i].T, dz)
            dB[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self._activate_derivative(self.z_values[i-1])

        return {'weights': dW, 'biases': dB}

    def update_weights(self, gradients):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * gradients['weights'][i]
            self.biases[i] -= self.learning_rate * gradients['biases'][i]

    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int)

    def compute_loss(self, X, y):
        output = self.forward(X)
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

    def compute_accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def train_step(self, X, y):
        output = self.forward(X)
        loss = self.compute_loss(X, y)
        gradients = self.backward(X, y, output)
        self.update_weights(gradients)
        return loss

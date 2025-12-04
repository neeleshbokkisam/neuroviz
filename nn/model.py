"""
Neural Network Implementation from Scratch using NumPy
Supports forward pass, backward pass, and stochastic gradient descent
"""

import numpy as np


class NeuralNetwork:
    """
    A fully connected neural network implemented from scratch.
    Supports multiple hidden layers, various activation functions, and SGD.
    """
    
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers specifying neurons per layer
                        e.g., [2, 4, 4, 1] for 2 input, two hidden layers of 4, 1 output
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            # Xavier/Glorot initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Storage for forward pass activations and gradients
        self.activations = []
        self.z_values = []  # Pre-activation values
        
    def _activate(self, x):
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            # Clip to avoid overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activate_derivative(self, x):
        """Compute derivative of activation function."""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Output predictions
        """
        # Store activations for backpropagation
        self.activations = [X]
        self.z_values = []
        
        current = X
        
        # Forward through hidden layers
        for i in range(self.num_layers - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current = self._activate(z)
            self.activations.append(current)
        
        # Output layer (sigmoid for binary classification)
        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid for output
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        """
        Backward pass (backpropagation) to compute gradients.
        
        Args:
            X: Input data
            y: True labels
            output: Network predictions from forward pass
        
        Returns:
            Dictionary with weight and bias gradients
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        dB = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (binary cross-entropy derivative)
        dz = output - y
        
        # Backpropagate through layers
        for i in range(self.num_layers - 1, -1, -1):
            # Gradient for weights and biases
            dW[i] = (1 / m) * np.dot(self.activations[i].T, dz)
            dB[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)
            
            # Propagate error to previous layer
            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self._activate_derivative(self.z_values[i-1])
        
        return {'weights': dW, 'biases': dB}
    
    def update_weights(self, gradients):
        """
        Update weights and biases using computed gradients (SGD).
        
        Args:
            gradients: Dictionary with 'weights' and 'biases' gradients
        """
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * gradients['weights'][i]
            self.biases[i] -= self.learning_rate * gradients['biases'][i]
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input data
        
        Returns:
            Binary predictions (0 or 1)
        """
        output = self.forward(X)
        return (output >= 0.5).astype(int)
    
    def compute_loss(self, X, y):
        """
        Compute binary cross-entropy loss.
        
        Args:
            X: Input data
            y: True labels
        
        Returns:
            Loss value
        """
        output = self.forward(X)
        # Clip to avoid log(0)
        output = np.clip(output, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        return loss
    
    def compute_accuracy(self, X, y):
        """
        Compute classification accuracy.
        
        Args:
            X: Input data
            y: True labels
        
        Returns:
            Accuracy (0-1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def train_step(self, X, y):
        """
        Perform one training step (forward + backward + update).
        
        Args:
            X: Input data
            y: True labels
        
        Returns:
            Loss value for this step
        """
        # Forward pass
        output = self.forward(X)
        
        # Compute loss
        loss = self.compute_loss(X, y)
        
        # Backward pass
        gradients = self.backward(X, y, output)
        
        # Update weights
        self.update_weights(gradients)
        
        return loss



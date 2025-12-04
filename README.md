# 🧠 NeuroViz: Neural Network Learning Visualization

An interactive Streamlit application that visualizes how a neural network learns to classify 2D data in real-time. Built entirely from scratch using NumPy (no PyTorch or TensorFlow).

## Features

- **Neural Network from Scratch**: Full implementation using only NumPy
  - Forward pass with configurable activation functions
  - Backward pass with manual gradient computation
  - Stochastic Gradient Descent (SGD) optimization

- **Interactive Visualizations**:
  - Real-time decision boundary evolution
  - Training loss curve
  - Training accuracy curve

- **Multiple Datasets**:
  - XOR dataset
  - Two Moons (make_moons)
  - Concentric Circles (make_circles)

- **Customizable Parameters**:
  - Number of hidden layers (1-5)
  - Neurons per hidden layer (2-20)
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Learning rate (0.001-1.0)
  - Number of training epochs (1-1000)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd neuroviz
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser. Use the sidebar to:
1. Select a dataset
2. Configure the network architecture
3. Set training parameters
4. Click "Train" to start training
5. Watch the decision boundary evolve in real-time!

## Project Structure

```
neuroviz/
├── nn/
│   └── model.py          # Neural network implementation
├── data/
│   └── datasets.py       # Dataset loaders
├── plots/
│   └── plot_utils.py     # Visualization utilities
├── app.py                # Streamlit frontend
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## How It Works

### Neural Network Architecture

The neural network is a fully connected (dense) network with:
- **Input Layer**: 2 neurons (for 2D data)
- **Hidden Layers**: Configurable number and size
- **Output Layer**: 1 neuron (binary classification with sigmoid activation)

### Training Process

1. **Forward Pass**: Data flows through the network, applying weights, biases, and activation functions
2. **Loss Computation**: Binary cross-entropy loss is calculated
3. **Backward Pass**: Gradients are computed manually using the chain rule
4. **Weight Update**: Weights and biases are updated using SGD

### Activation Functions

- **ReLU**: `f(x) = max(0, x)` - Fast convergence, may suffer from dying ReLU
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))` - Smooth gradients, can saturate
- **Tanh**: `f(x) = tanh(x)` - Zero-centered, similar to sigmoid


## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.



"""
Streamlit app for visualizing neural network learning in real-time.
"""

import streamlit as st
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nn.model import NeuralNetwork
from data.datasets import load_dataset, normalize_data
from plots.plot_utils import plot_decision_boundary, plot_training_metrics


# Page configuration
st.set_page_config(
    page_title="NeuroViz - Neural Network Visualization",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 NeuroViz: Neural Network Learning Visualization")
st.markdown("**Watch a neural network learn in real-time!** This app demonstrates how neural networks learn to classify 2D data using only NumPy.")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = []
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'X_norm' not in st.session_state:
    st.session_state.X_norm = None
if 'X_min' not in st.session_state:
    st.session_state.X_min = None
if 'X_max' not in st.session_state:
    st.session_state.X_max = None
if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
if 'training' not in st.session_state:
    st.session_state.training = False
if 'arch_params' not in st.session_state:
    st.session_state.arch_params = {'num_layers': 0, 'neurons': 0}


# Sidebar for controls
st.sidebar.header("⚙️ Network Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Dataset",
    options=['xor', 'moons', 'circles'],
    index=0,
    help="Choose a 2D classification dataset"
)

# Network architecture
st.sidebar.subheader("Architecture")
num_hidden_layers = st.sidebar.slider(
    "Number of Hidden Layers",
    min_value=1,
    max_value=5,
    value=2,
    help="Number of hidden layers in the network"
)

neurons_per_layer = st.sidebar.slider(
    "Neurons per Hidden Layer",
    min_value=2,
    max_value=20,
    value=4,
    help="Number of neurons in each hidden layer"
)

# Training parameters
st.sidebar.subheader("Training Parameters")
activation = st.sidebar.selectbox(
    "Activation Function",
    options=['relu', 'sigmoid', 'tanh'],
    index=0,
    help="Activation function for hidden layers"
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    min_value=0.001,
    max_value=1.0,
    value=0.1,
    step=0.001,
    format="%.3f",
    help="Learning rate for gradient descent"
)

num_epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=1,
    max_value=1000,
    value=100,
    help="Number of training epochs"
)

# Buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    train_button = st.button("🚀 Train", use_container_width=True)

with col2:
    reset_button = st.button("🔄 Reset", use_container_width=True)


# Load dataset
@st.cache_data
def get_dataset(name):
    """Load and cache dataset."""
    X, y = load_dataset(name, n_samples=200, noise=0.1)
    X_norm, X_min, X_max = normalize_data(X)
    return X, y, X_norm, X_min, X_max


# Handle reset button
if reset_button:
    st.session_state.model = None
    st.session_state.loss_history = []
    st.session_state.accuracy_history = []
    st.session_state.epoch = 0
    st.session_state.training = False
    st.session_state.arch_params = {'num_layers': 0, 'neurons': 0}
    st.rerun()


# Load dataset
X, y, X_norm, X_min, X_max = get_dataset(dataset_name)
st.session_state.X = X
st.session_state.y = y
st.session_state.X_norm = X_norm
st.session_state.X_min = X_min
st.session_state.X_max = X_max


# Check if architecture changed
arch_changed = (
    st.session_state.arch_params['num_layers'] != num_hidden_layers or
    st.session_state.arch_params['neurons'] != neurons_per_layer
)

# Initialize or recreate model if needed
if st.session_state.model is None or arch_changed:
    # Build layer sizes
    layer_sizes = [2]  # Input layer (2D data)
    for _ in range(num_hidden_layers):
        layer_sizes.append(neurons_per_layer)
    layer_sizes.append(1)  # Output layer (binary classification)
    
    # Create model
    st.session_state.model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=activation,
        learning_rate=learning_rate
    )
    st.session_state.loss_history = []
    st.session_state.accuracy_history = []
    st.session_state.epoch = 0
    st.session_state.arch_params = {
        'num_layers': num_hidden_layers,
        'neurons': neurons_per_layer
    }

# Update model parameters if changed (but architecture is same)
elif (st.session_state.model.activation != activation or 
      st.session_state.model.learning_rate != learning_rate):
    st.session_state.model.activation = activation
    st.session_state.model.learning_rate = learning_rate


# Handle training
if train_button:
    st.session_state.training = True
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training step
        loss = st.session_state.model.train_step(X_norm, y)
        accuracy = st.session_state.model.compute_accuracy(X_norm, y)
        
        # Store history
        st.session_state.loss_history.append(loss)
        st.session_state.accuracy_history.append(accuracy)
        st.session_state.epoch += 1
        
        # Update progress
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    
    st.session_state.training = False
    progress_bar.empty()
    status_text.empty()
    st.rerun()


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Decision Boundary")
    
    # Plot decision boundary
    if st.session_state.model is not None:
        # Use original data for visualization
        X_vis = st.session_state.X
        fig_boundary = plot_decision_boundary(
            st.session_state.model,
            X_vis,
            st.session_state.y,
            X_min=st.session_state.X_min,
            X_max=st.session_state.X_max
        )
        st.plotly_chart(fig_boundary, use_container_width=True)
    else:
        st.info("Click 'Train' to start training the neural network!")

with col2:
    st.subheader("Network Info")
    
    if st.session_state.model is not None:
        st.write(f"**Architecture:**")
        layer_sizes_str = " → ".join(map(str, st.session_state.model.layer_sizes))
        st.code(layer_sizes_str)
        
        st.write(f"**Activation:** {st.session_state.model.activation.upper()}")
        st.write(f"**Learning Rate:** {st.session_state.model.learning_rate:.3f}")
        st.write(f"**Total Epochs:** {st.session_state.epoch}")
        
        if len(st.session_state.loss_history) > 0:
            st.write(f"**Current Loss:** {st.session_state.loss_history[-1]:.4f}")
            st.write(f"**Current Accuracy:** {st.session_state.accuracy_history[-1]:.4f}")


# Training metrics
if len(st.session_state.loss_history) > 0:
    st.subheader("Training Metrics")
    fig_metrics = plot_training_metrics(
        st.session_state.loss_history,
        st.session_state.accuracy_history
    )
    st.plotly_chart(fig_metrics, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(
    """
    **About NeuroViz:** This educational tool demonstrates neural network learning from scratch using only NumPy. 
    Watch how the decision boundary evolves as the network learns to classify 2D data!
    """
)


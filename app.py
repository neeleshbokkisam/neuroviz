import streamlit as st
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nn.model import NeuralNetwork
from data.datasets import load_dataset, normalize_data
from plots.plot_utils import plot_decision_boundary, plot_training_metrics


st.set_page_config(page_title="neuroviz", layout="wide")

st.title("neuroviz")

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


st.sidebar.header("config")

dataset_name = st.sidebar.selectbox("dataset", ['xor', 'moons', 'circles'])

st.sidebar.subheader("architecture")
num_hidden_layers = st.sidebar.slider("hidden layers", 1, 5, 2)
neurons_per_layer = st.sidebar.slider("neurons per layer", 2, 20, 4)

st.sidebar.subheader("training")
activation = st.sidebar.selectbox("activation", ['relu', 'sigmoid', 'tanh'])
learning_rate = st.sidebar.slider("learning rate", 0.001, 1.0, 0.1, step=0.001, format="%.3f")
num_epochs = st.sidebar.slider("epochs", 1, 1000, 100)

col1, col2 = st.sidebar.columns(2)
with col1:
    train_button = st.button("train", use_container_width=True)
with col2:
    reset_button = st.button("reset", use_container_width=True)


@st.cache_data
def get_dataset(name):
    X, y = load_dataset(name, n_samples=200, noise=0.1)
    X_norm, X_min, X_max = normalize_data(X)
    return X, y, X_norm, X_min, X_max


if reset_button:
    st.session_state.model = None
    st.session_state.loss_history = []
    st.session_state.accuracy_history = []
    st.session_state.epoch = 0
    st.session_state.training = False
    st.session_state.arch_params = {'num_layers': 0, 'neurons': 0}
    st.rerun()


X, y, X_norm, X_min, X_max = get_dataset(dataset_name)
st.session_state.X = X
st.session_state.y = y
st.session_state.X_norm = X_norm
st.session_state.X_min = X_min
st.session_state.X_max = X_max


arch_changed = (
    st.session_state.arch_params['num_layers'] != num_hidden_layers or
    st.session_state.arch_params['neurons'] != neurons_per_layer
)

if st.session_state.model is None or arch_changed:
    layer_sizes = [2]
    for _ in range(num_hidden_layers):
        layer_sizes.append(neurons_per_layer)
    layer_sizes.append(1)

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

elif (st.session_state.model.activation != activation or
      st.session_state.model.learning_rate != learning_rate):
    st.session_state.model.activation = activation
    st.session_state.model.learning_rate = learning_rate


if train_button:
    st.session_state.training = True
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(num_epochs):
        loss = st.session_state.model.train_step(X_norm, y)
        accuracy = st.session_state.model.compute_accuracy(X_norm, y)

        st.session_state.loss_history.append(loss)
        st.session_state.accuracy_history.append(accuracy)
        st.session_state.epoch += 1

        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f"epoch {epoch + 1}/{num_epochs}  loss {loss:.4f}  acc {accuracy:.4f}")

    st.session_state.training = False
    progress_bar.empty()
    status_text.empty()
    st.rerun()


col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("decision boundary")

    if st.session_state.model is not None:
        fig_boundary = plot_decision_boundary(
            st.session_state.model,
            st.session_state.X,
            st.session_state.y,
            X_min=st.session_state.X_min,
            X_max=st.session_state.X_max
        )
        st.plotly_chart(fig_boundary, use_container_width=True)
    else:
        st.write("hit train")

with col2:
    st.subheader("model")

    if st.session_state.model is not None:
        layer_sizes_str = " → ".join(map(str, st.session_state.model.layer_sizes))
        st.code(layer_sizes_str)
        st.write(f"{st.session_state.model.activation}, lr={st.session_state.model.learning_rate:.3f}")
        st.write(f"epochs run: {st.session_state.epoch}")

        if st.session_state.loss_history:
            st.write(f"loss: {st.session_state.loss_history[-1]:.4f}")
            st.write(f"acc: {st.session_state.accuracy_history[-1]:.4f}")


if st.session_state.loss_history:
    st.subheader("metrics")
    fig_metrics = plot_training_metrics(
        st.session_state.loss_history,
        st.session_state.accuracy_history
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

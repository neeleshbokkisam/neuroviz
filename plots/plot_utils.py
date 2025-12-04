"""
Plotting utilities for visualizing neural network training.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_decision_boundary(model, X, y, resolution=100, X_min=None, X_max=None):
    """
    Plot decision boundary and data points.
    
    Args:
        model: Trained neural network model
        X: Input features (original scale, for visualization)
        y: True labels
        resolution: Resolution of the mesh grid
        X_min: Minimum values for normalization (if model was trained on normalized data)
        X_max: Maximum values for normalization (if model was trained on normalized data)
    
    Returns:
        Plotly figure object
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Predict on mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Normalize grid points if normalization parameters provided
    if X_min is not None and X_max is not None:
        grid_points_norm = (grid_points - X_min) / (X_max - X_min + 1e-8)
    else:
        grid_points_norm = grid_points
    
    Z = model.forward(grid_points_norm)
    Z = Z.reshape(xx.shape)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add decision boundary contour
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z,
        colorscale='RdYlBu',
        showscale=True,
        opacity=0.6,
        name='Decision Boundary'
    ))
    
    # Add data points
    class_0 = X[y.flatten() == 0]
    class_1 = X[y.flatten() == 1]
    
    if len(class_0) > 0:
        fig.add_trace(go.Scatter(
            x=class_0[:, 0],
            y=class_0[:, 1],
            mode='markers',
            marker=dict(color='blue', size=8, symbol='circle'),
            name='Class 0',
            showlegend=True
        ))
    
    if len(class_1) > 0:
        fig.add_trace(go.Scatter(
            x=class_1[:, 0],
            y=class_1[:, 1],
            mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            name='Class 1',
            showlegend=True
        ))
    
    fig.update_layout(
        title='Decision Boundary',
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        width=600,
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_loss_curve(loss_history):
    """
    Plot training loss curve.
    
    Args:
        loss_history: List of loss values over training
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(loss_history))),
        y=loss_history,
        mode='lines',
        name='Training Loss',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title='Training Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        width=600,
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_accuracy_curve(accuracy_history):
    """
    Plot training accuracy curve.
    
    Args:
        accuracy_history: List of accuracy values over training
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(accuracy_history))),
        y=accuracy_history,
        mode='lines',
        name='Training Accuracy',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Training Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        width=600,
        height=400,
        yaxis=dict(range=[0, 1.05]),
        hovermode='x unified'
    )
    
    return fig


def plot_training_metrics(loss_history, accuracy_history):
    """
    Plot both loss and accuracy in subplots.
    
    Args:
        loss_history: List of loss values
        accuracy_history: List of accuracy values
    
    Returns:
        Plotly figure object with subplots
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Loss', 'Training Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(loss_history))),
            y=loss_history,
            mode='lines',
            name='Loss',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(accuracy_history))),
            y=accuracy_history,
            mode='lines',
            name='Accuracy',
            line=dict(color='green', width=2)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=1, col=2)
    
    fig.update_layout(
        title_text='Training Metrics',
        width=1200,
        height=400,
        showlegend=True
    )
    
    return fig


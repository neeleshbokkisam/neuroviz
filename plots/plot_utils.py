import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_decision_boundary(model, X, y, resolution=100, X_min=None, X_max=None):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if X_min is not None and X_max is not None:
        grid_points_norm = (grid_points - X_min) / (X_max - X_min + 1e-8)
    else:
        grid_points_norm = grid_points

    Z = model.forward(grid_points_norm).reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=Z,
        colorscale='RdYlBu',
        showscale=True,
        opacity=0.6,
    ))

    class_0 = X[y.flatten() == 0]
    class_1 = X[y.flatten() == 1]

    if len(class_0) > 0:
        fig.add_trace(go.Scatter(
            x=class_0[:, 0],
            y=class_0[:, 1],
            mode='markers',
            marker=dict(color='blue', size=6),
            name='0',
        ))

    if len(class_1) > 0:
        fig.add_trace(go.Scatter(
            x=class_1[:, 0],
            y=class_1[:, 1],
            mode='markers',
            marker=dict(color='red', size=6, symbol='x'),
            name='1',
        ))

    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        height=450,
        hovermode='closest',
        margin=dict(l=40, r=40, t=20, b=40),
    )

    return fig


def plot_loss_curve(loss_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(loss_history))),
        y=loss_history,
        mode='lines',
        line=dict(width=1.5),
    ))
    fig.update_layout(xaxis_title='epoch', yaxis_title='loss', height=350)
    return fig


def plot_accuracy_curve(accuracy_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(accuracy_history))),
        y=accuracy_history,
        mode='lines',
        line=dict(width=1.5),
    ))
    fig.update_layout(xaxis_title='epoch', yaxis_title='acc', height=350, yaxis=dict(range=[0, 1.05]))
    return fig


def plot_training_metrics(loss_history, accuracy_history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('loss', 'acc'))

    fig.add_trace(
        go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode='lines', line=dict(width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(accuracy_history))), y=accuracy_history, mode='lines', line=dict(width=1.5)),
        row=1, col=2
    )

    fig.update_xaxes(title_text="epoch", row=1, col=1)
    fig.update_xaxes(title_text="epoch", row=1, col=2)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_yaxes(title_text="acc", range=[0, 1.05], row=1, col=2)

    fig.update_layout(height=350, showlegend=False, margin=dict(l=40, r=40, t=40, b=40))

    return fig

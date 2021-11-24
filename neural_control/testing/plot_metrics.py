import argparse
from itertools import cycle
import json
import os
from InputsManager import InputsManager
import plotly.express as px
from PIL import ImageColor
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
colors = cycle(px.colors.qualitative.G10)


def plot_error(mean: np.array, sigma: np.array, color: str, name: str, fig: go.Figure = None):
    """
    Plot mean with fillled opaque area by standard deviation sigma

    Params:
        mean: mean value
        sigma: standard deviation
        color: color of the lines
        name: name of the graph
        fig: figure to plot on

    Returns:
        fig: figure with plot

    """
    if not fig: fig = go.Figure()
    x = np.arange(len(mean))
    color_rgb = ImageColor.getcolor(color, "RGB")
    color_rgba = color_rgb + (0.15,)  # TODO
    color_rgb = f'rgb' + str(color_rgb)
    color_rgba = f'rgba' + str(color_rgba)
    # Mean
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            name=name,
            line=dict(color=color_rgb),
        )
    )
    # Standard deviation
    fig.add_trace(
        go.Scatter(
            x=x.tolist() + x.tolist()[::-1],
            y=(mean + sigma).tolist() + (mean - sigma).tolist()[::-1],
            fill='toself',
            fillcolor=color_rgba,
            line=dict(color=color_rgba),
            hoverinfo='skip',
            showlegend=False,
        )
    )
    # Create line on y=0
    fig.add_trace(
        go.Scatter(
            x=[0, np.size(mean)],
            y=[0, 0],
            mode='lines',
            line=dict(
                color='rgb(0,0,0)',
                dash='dash'),
            showlegend=False,
        )
    )
    # fig.show()
    # fig.update_yaxes(type="log")
    return fig


def plot_point_error(error: float, color: str, name: str, sigma: float = None, fig: go.Figure = None):
    """
    Plot point error with error band on the y axis

    Params:
        error: point error
        sigma: standard deviation
        color: color of the error band
        title: title of the graph
        fig: figure to plot on

    Returns:
        fig: figure with plot

    """
    if not fig: fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[error],
            line=dict(color=color, dash='dash'),
            # error_y=dict(
            #     type='data',
            #     array=np.array(sigma),
            #     visible=True,
            #     color=color,
            #     thickness=1.5,
            #     width=2,
            # ),
            name=name,
            showlegend=False,
        )
    )
    # fig.show()
    return fig


if __name__ == '__main__':
    pio.templates.default = "plotly_white"
    fig, axes = [], []
    for i in range(4): fig.append(go.Figure())
    parser = argparse.ArgumentParser(description='Calculate metrics from test simulations')
    parser.add_argument("run_paths", nargs="+", help="Paths to folders containing model data")
    args = parser.parse_args()
    runs_paths = args.run_paths
    for run_path in runs_paths:
        # run_path = root + run_folder
        # Pre process variables
        try:
            inp = InputsManager(os.path.abspath(f"{run_path}/inputs.json"))
        except:
            print(f"Could not load information from {run_path}")
            continue
        tests = os.listdir(os.path.abspath(run_path + "/tests/"))
        tests = [test for test in tests if "test1" in test]  # TODO only calculating metrics for one test
        for i, test in enumerate(tests):
            # Load metrics json
            datapath = f"{run_path}/tests/{test}/"
            with open(f"{datapath}/metrics.json", "r") as f:
                metrics = json.load(f)
            # Plot general error
            color = next(colors)
            plot_error(
                np.array(metrics["general_error_xy"]),
                np.array(metrics["general_error_xy_stdd"]),
                color,
                run_path.split("/")[-2],
                fig[0])

            # Force variation
            fig[1].add_trace(
                go.Bar(
                    x=(i,),
                    y=(metrics["force_change"],),
                    name=run_path.split("/")[-2],
                    marker_color=color,
                ))
            # Force mean
            fig[2].add_trace(
                go.Bar(
                    x=(i,),
                    y=(metrics["force_mean_mean"],),
                    name=run_path.split("/")[-2],
                    marker_color=color,
                ))
            # Force max value
            fig[3].add_trace(
                go.Bar(
                    x=(i,),
                    y=(metrics["force_mean_max_value"],),
                    name=run_path.split("/")[-2],
                    marker_color=color,
                ))
    fig[0].update_layout(title="Trajectory Error", yaxis_title='Error')
    fig[1].update_layout(yaxis_title='Force variation')
    fig[2].update_layout(yaxis_title='Force mean')
    fig[3].update_layout(yaxis_title='Force max')
    # Show plots
    for i, fig_ in enumerate(fig):
        fig_.write_html(f"/home/ramos/phiflow/neural_control/testing/fig{i}.html")

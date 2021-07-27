from cProfile import label
import json
from PIL import ImageColor
import plotly.express as px
import collections
import plotly.graph_objects as go
import plotly.io as pio
from itertools import cycle
import numpy as np
from InputsManager import InputsManager
import os
from natsort import natsorted
import argparse
import matplotlib.pyplot as plt
from Logger import Logger
from traitlets.traitlets import default
# colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
colors = cycle(px.colors.qualitative.G10)


def calculate_error(x: np.array, y: np.array = 0, normalizing_factor: float = 1):
    """
    Calculate mena spatial error from x and y components and its standard deviation

    Params:
        x: x component of error
        y: y component of error
        normalizing_factor: normalizing factor

    Returns:
        (error,sigma): mean spatial error xy and standard deviation
        labels: labels of outputs

    """
    error = np.sqrt(x**2 + y**2) / normalizing_factor
    sigma = np.std(error, axis=0)
    error = np.mean(error, axis=0)
    return (error, sigma), ('', '_stdd')


def calculate_stopping_error(x: np.array, y: np.array = 0, normalizing_factor: float = 1):
    """
    Mean error of the last 15% of the trajectory

    Params:
        x: x component of error
        y: y component of error
        normalizing_factor: normalizing factor

    Returns:
        (last_mean_error, last_sigma) : spatial error xy and its standard deviation
        labels: labels of outputs

    """
    error = np.sqrt(x**2 + y**2) / normalizing_factor
    sigma = np.std(error, axis=0)
    mean_error = np.mean(error, axis=0)
    index = int(len(mean_error) * 0.15)
    last_mean_error = np.mean(mean_error[index:])
    last_sigma = np.mean(sigma[index:])
    return (last_mean_error, last_sigma), ('', '_stdd')


def calculate_mean(x: np.array, y: np.array = 0, normalizing_factor: float = 1):
    """
    Integrate coordinates x and y

    Params:
        x: x component
        y: y component
        normalizing_factor: normalizing factor

    Returns:
        (mean, max_value): mean value and max value of mean values
        labels: labels of outputs

    """
    value = np.sqrt((x / normalizing_factor)**2 + (y / normalizing_factor)**2)
    max_value = np.amax(value)
    mean = np.mean(value)
    return (mean, max_value), ('_mean', '_max_value')


def plot_error(mean: np.array, sigma: np.array, color: str, title: str, fig: go.Figure = None):
    """
    Plot mean with fillled opaque area by standard deviation sigma

    Params:
        mean: mean value
        sigma: standard deviation
        color: color of the lines
        title: title of the graph
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
    fig = go.Figure()
    # Mean
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            name=f"Mean error",
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
    fig.update_layout(title=title, yaxis_title='Error')
    # fig.show()
    return fig


def plot_point_error(error: float, sigma: float, color: str, title: str, fig: go.Figure = None):
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
            error_y=dict(
                type='data',
                array=np.array(sigma),
                visible=True,
                color=color,
                thickness=1.5,
                width=2,
            ),
            showlegend=False,
        )
    )
    fig.update_layout(title=title, yaxis_title='Error')
    # fig.show()
    return fig


if __name__ == '__main__':
    pio.templates.default = "plotly_white"
    metrics = {
        'general_error_xy': {
            'vars': ['error_x', 'error_y'],
            'func': calculate_error,
        },
        'stopping_error_xy': {
            'vars': ['error_x', 'error_y'],
            'func': calculate_stopping_error,
        },
        'general_error_ang': {
            'vars': ['error_ang'],
            'func': calculate_error,
        },
        'stopping_error_ang': {
            'vars': ['error_ang'],
            'func': calculate_stopping_error,
        },
        'control_force': {
            'vars': ['control_force_x', 'control_force_y'],
            'func': calculate_mean,
        },
        'control_torque': {
            'vars': ['control_torque'],
            'func': calculate_mean,
        }
    }
    fig, axes = [], []
    for i in range(2): fig.append(go.Figure())
    parser = argparse.ArgumentParser(description='Calculate metrics from test simulations')
    parser.add_argument("root_directory", help="Root directory that contain runs")
    parser.add_argument("runs_directories", nargs="+", help="Folders containing model data")
    args = parser.parse_args()
    root = args.root_directory
    runs_folders = args.runs_directories
    for run_folder in runs_folders:
        run_path = root + run_folder
        # Pre process variables
        try:
            inp = InputsManager(f"{run_path}/inputs.json")
        except:
            print(f"Could not load information from {run_path}")
            continue
        tests = os.listdir(run_path + "/tests/")
        for test in tests:
            datapath = f"{run_path}/tests/{test}/data/"
            id = run_path.split("/")[-1]
            # color = next(colors)
            # color_rgb = ImageColor.getcolor(color, "RGB")
            # if should_log_wandb:
            #     os.environ["WANDB_DIR"] = run_path
            #     os.environ["WANDB_MODE"] = "run"  # Run wandb online so we can sync with previous run
            #     logger = Logger(run_path)
            #     logger.sync_wandb()
            #     # Resume from previous run
            #     logger.initialize_wandb(
            #         run_path.split('/')[-1],
            #         "run",
            #         project="neural_controller_translation_only" if inp.translation_only else "neural_controller",
            #         resume="must",
            #         id=inp.id)
            # Get files, cases and snapshots
            all_files = os.listdir(f"{datapath}/error_x/")
            all_files = [file.split("error_x")[1] for file in all_files]  # Remove prefix
            cases = natsorted(tuple(set([file.split("case")[1][:4] for file in all_files])))
            snapshots = natsorted(tuple(set((file.split("_")[-1][:4] for file in all_files))))
            norm_vars = dict(
                error_x=inp.simulation['obs_width'],
                error_y=inp.simulation['obs_width'],
                control_force_x=inp.simulation['obs_mass'] * inp.max_acc,
                control_force_y=inp.simulation['obs_mass'] * inp.max_acc,
                error_ang=np.pi,
                control_torque=inp.simulation['obs_inertia'] * inp.max_ang_acc,
            )
            # Loop through metrics
            export_dict = {}
            for metric_name, attrs in metrics.items():
                vars = attrs['vars']
                func = attrs['func']
                values_all = collections.defaultdict(list)
                # Load data
                for var in vars:
                    for case in cases:
                        try:
                            values = np.array([np.load(f"{datapath}/{var}/{var}_case{case}_{snapshot}.npy") for snapshot in snapshots])
                        except:
                            print(f"Could not load {var} for case {case}")
                            continue
                        values_all[var] += [values]
                    values_all[var] = np.array(values_all[var])
                # Calculate metric
                metrics_values, metrics_labels = func(*values_all.values(), normalizing_factor=norm_vars[var])
                # Logging
                for metric_values, metric_label in zip(metrics_values, metrics_labels):
                    export_dict[f"{metric_name}{metric_label}"] = metric_values.tolist()  # Local logging
            # Export metrics to json file
            with open(f"{inp.export_path}/tests/{test}/metrics.json", 'w') as f:
                json.dump(export_dict, f, indent="    ")

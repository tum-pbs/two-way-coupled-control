from cProfile import label
import json
import collections
import numpy as np
from InputsManager import InputsManager
import os
from natsort import natsorted
import argparse
import matplotlib.pyplot as plt
# colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def calculate_variability(y: np.array, y_additional: np.array = 0, normalizing_factor: float = 1):
    """
    Calculate how much y varies between consecutive points on average

    Params:
        y: y component
        normalizing_factor: normalizing factor

    Returns:
    """
    values = np.sqrt(y**2 + y_additional**2) / normalizing_factor
    return (np.mean(np.abs(values[1:] - values[:-1])),), ('',)


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
    index = int(len(mean_error) * 0.85)
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


if __name__ == '__main__':
    metrics = {
        'general_error_xy': {
            'vars': ['error_x', 'error_y'],
            'func': calculate_error,
        },
        'stopping_error_xy': {
            'vars': ['error_x', 'error_y'],
            'func': calculate_stopping_error,
        },
        'force': {
            'vars': ['control_force_x', 'control_force_y'],
            'func': calculate_error,
        },
        # 'torque_change': {
        #     'vars': ['control_torque'],
        #     'func': calculate_variability,
        # },
        # 'torque_mean': {
        #     'vars': ['control_torque'],
        #     'func': calculate_mean,
        # },
        # 'general_error_ang': {
        #     'vars': ['error_ang'],
        #     'func': calculate_error,
        # },
        # 'stopping_error_ang': {
        #     'vars': ['error_ang'],
        #     'func': calculate_stopping_error,
        # },
    }
    fig, axes = [], []
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
        tests = [test for test in tests if test.split("_")[0] in ["test1", "test2"]]  # TODO only calculating metrics for one test
        for test in tests:
            datapath = f"{run_path}/tests/{test}/data/"
            id = run_path.split("/")[-1]
            # Get files, cases and snapshots
            all_files = os.listdir(os.path.abspath(f"{datapath}/error_x/"))
            all_files = [file.split("error_x")[1] for file in all_files]  # Remove prefix
            cases = natsorted(tuple(set([file.split("case")[1][:4] for file in all_files])))
            snapshots = natsorted(tuple(set((file.split("_")[-1][:5] for file in all_files))))
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
                            values = np.array([np.load(os.path.abspath(f"{datapath}/{var}/{var}_case{case}_{snapshot}.npy")) for snapshot in snapshots])
                        except:
                            print(f"Could not load {var} for case {case}")
                            continue
                        values_all[var] += [values]
                    values_all[var] = np.array(values_all[var])
                # Calculate metric
                metrics_values, metrics_labels = func(*values_all.values(), )
                # Logging
                for metric_values, metric_label in zip(metrics_values, metrics_labels):
                    export_dict[f"{metric_name}{metric_label}"] = metric_values.flatten().tolist()  # Local logging
            # Export metrics to json file
            with open(os.path.abspath(f"{run_path}/tests/{test}/metrics.json"), 'w') as f:
                json.dump(export_dict, f, indent="    ")
    print("Done")

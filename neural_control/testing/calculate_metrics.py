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
    Mean error of the last 40% of the trajectory

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
    index = int(len(mean_error) * 0.40)
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


def remove_repeated(x: np.array, y: np.array = None, normalizing_factor: float = 1):
    """
    Remove repeated entries on time axis.
    x and y must be rank 3 with dimensions (batch, time, coordinates)
    """
    if y is not None: x = np.concatenate((x, y), axis=-1)
    # Find unique elements of each column in x
    all_objectives = []
    for i, rows in enumerate(x):
        objectives = rows[None, 0]
        for column in rows:
            res = column - objectives[-1].squeeze()
            if any(np.abs(res) > 1e-6):
                objectives = np.concatenate((objectives, column[None]))
        all_objectives.append(objectives)
    out = np.swapaxes(all_objectives, 1, 2)
    return (out,), ('',)


def execute(run_path):
    """
    Calculate all metrics
    """
    metrics = dict(
        general_error_xy=dict(
            vars=['error_x', 'error_y'],
            func=calculate_error,
        ),
        stopping_error_xy=dict(
            vars=['error_x', 'error_y'],
            func=calculate_stopping_error,
        ),
        force=dict(
            vars=['control_force_x', 'control_force_y'],
            func=calculate_error,
        ),
        trajectory=dict(
            vars=["obs_xy"],
            func=lambda xy: ((xy.swapaxes(1, 2),), ('',)),
        ),
        objective_xy=dict(
            vars=["reference_x", "reference_y"],
            func=remove_repeated
        )

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
    )

    # run_path = root + run_folder
    # Pre process variables
    tests = os.listdir(os.path.abspath(run_path + "/tests/"))
    tests = [test for test in tests if test.split("_")[0] in ["test1", "test2", "test3"]]  # TODO only calculating metrics for one test
    for test in tests:
        datapath = f"{run_path}/tests/{test}/data/"
        # Get files cases
        all_files = os.listdir(os.path.abspath(f"{datapath}/error_x/"))
        all_files = [file.split("error_x")[1] for file in all_files]  # Remove prefix
        cases = natsorted(tuple(set([file.split("case")[1][:4] for file in all_files])))
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
                        values = np.load(os.path.abspath(f"{datapath}/{var}/{var}_case{case}.npy"))
                    except FileNotFoundError:
                        print(f"\n Could not load {var} for case {case} in {datapath}. Make sure frames are grouped correctly. \n")
                        exit()
                    values_all[var] += [values]
                values_all[var] = np.array(values_all[var])
            # Calculate metric
            values, labels = func(*values_all.values(), )
            # Logging
            for value, label in zip(values, labels):
                export_dict[f"{metric_name}{label}"] = value.tolist()  # Local logging
        # Export metrics to json file
        with open(os.path.abspath(f"{run_path}/tests/{test}/metrics.json"), 'w') as f:
            json.dump(export_dict, f, indent="    ")
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics from test simulations')
    parser.add_argument("run_path", help="Paths to folders containing model data")
    args = parser.parse_args()
    runs_path = args.run_path
    execute(runs_path)

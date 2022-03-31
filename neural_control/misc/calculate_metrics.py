from cProfile import label
import json
import collections
import numpy as np
import os
from natsort import natsorted
import argparse
# import matplotlib.pyplot as plt
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


def calculate_norm(x: np.array, y: np.array = 0, normalizing_factor: float = 1):
    """
    Calculate mean norm from x and y components and its standard deviation

    Params:
        x: x component
        y: y component
        normalizing_factor: normalizing factor

    Returns:
        (z,sigma): mean of x y and standard deviation
        labels: labels of outputs

    """
    z = np.sqrt(x**2 + y**2) / normalizing_factor
    sigma = np.std(z, axis=0)
    z = np.mean(z, axis=0)
    return (z, sigma), ('', '_stdd')


def calculate_norm_individual(x: np.array, y: np.array = 0, normalizing_factor: float = 1):
    """
    Calculate norm from x and y components

    Params:
        x: x component
        y: y component
        normalizing_factor: normalizing factor

    Returns:
        z: norm and standard deviation of all simulations
        labels: labels of outputs

    """
    z = np.sqrt(x**2 + y**2) / normalizing_factor
    return (z,), ('',)


def calculate_ss(x: np.array, reference_x: np.array, y: np.array = 0, reference_y: np.array = None, normalizing_factor: float = 1):
    """
    Steady steady error calculated by averaging error without considering the first
    25% of the trajectory after an objective is set.

    Params:
        x: x component of error
        y: y component of error
        normalizing_factor: normalizing factor

    Returns:
        (last_mean_error, last_sigma) : spatial error xy and its standard deviation
        labels: labels of outputs

    """
    if reference_y is None: objectives = reference_x
    else: objectives = np.concatenate((reference_x, reference_y), axis=-1)
    # Find number of sections in which a new objective is set
    objective_changes = np.linalg.norm(objectives[:, 1:, :] - objectives[:, :-1:], axis=-1) > 1e-6
    n_sections = 1
    n_sections += np.size(np.where(objective_changes[0] == 1))
    # Calculate the size of chunk of data that will be discarded
    ss_i = 0.25
    deleted_chunk_size = int(x.shape[1] / n_sections * ss_i)
    section_size = int(x.shape[1] / n_sections)
    delete_i = [k * section_size for k in range(n_sections)]
    delete_i_all = []
    for k in delete_i:
        for l in range(deleted_chunk_size):
            delete_i_all.append(k + l)
    z = np.sqrt(x**2 + y**2) / normalizing_factor
    z = np.delete(z, delete_i_all, axis=1)
    mean = np.mean(z, axis=0)
    # Average all tests
    std = np.std(mean)
    mean = np.mean(mean)
    return (mean, std), ('', '_stdd')


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


def execute(run_path, metrics_keys=None, tests=None, rotation_metrics=True):
    """
    Calculate all metrics
    """
    metrics = dict(
        error_xy=dict(
            vars=['error_x', 'error_y'],
            func=calculate_norm,
        ),
        error_xy_all=dict(
            vars=['error_x', 'error_y'],
            func=calculate_norm_individual,
        ),
        error_ang=dict(
            vars=['error_ang'],
            func=calculate_norm,
        ),
        error_ang_all=dict(
            vars=['error_ang'],
            func=calculate_norm_individual,
        ),
        ss_error_xy=dict(
            vars=['error_x', "reference_x", 'error_y', "reference_y"],
            func=calculate_ss,
        ),
        ss_error_ang=dict(
            vars=['error_ang', "reference_ang"],
            func=calculate_ss,
        ),
        forces_norm=dict(
            vars=['control_force_x', 'control_force_y'],
            # func=lambda xy: ((xy.swapaxes(1, 2),), ('',)),
            func=calculate_norm,
        ),
        trajectory=dict(
            vars=["obs_xy"],
            func=lambda xy: ((xy.swapaxes(1, 2),), ('',)),
        ),
        objective_xy=dict(
            vars=["reference_x", "reference_y"],
            func=remove_repeated
        ),
        angle=dict(
            vars=["obs_ang"],
            func=lambda alpha: ((-alpha.swapaxes(1, 2),), ('',)),
        ),
        torque_norm=dict(
            vars=["control_torque"],
            func=calculate_norm,
        ),
        objective_angle=dict(
            vars=["reference_ang"],
            func=remove_repeated
        ),
    )

    if not metrics_keys: metrics_keys = list(metrics.keys())
    # Gather test folders
    tests_folders = os.listdir(os.path.abspath(run_path + "/tests/"))
    if tests:
        tests = [test for test in tests_folders if any([(test_ in test) for test_ in tests])]
    else:
        tests = os.listdir(os.path.abspath(run_path + "/tests/"))
        tests = [test for test in tests if 'test' in test]
    # Loop through tests and calculate metrics
    for test in tests:
        datapath = f"{run_path}/tests/{test}/data/"
        # Get files cases
        all_files = os.listdir(os.path.abspath(f"{datapath}/error_x/"))
        all_files = [file.split("error_x")[1] for file in all_files]  # Remove prefix
        cases = natsorted(tuple(set([file.split("case")[1][:4] for file in all_files])))
        # Loop through metrics
        export_dict = {}
        for metric_name in metrics_keys:
            if not rotation_metrics and ('angle' in metric_name or 'torque' in metric_name):
                continue
            attrs = metrics[metric_name]
            vars = attrs['vars']
            func = attrs['func']
            values_all = collections.defaultdict(list)
            for var in vars:
                for case in cases:
                    try:
                        values = np.load(os.path.abspath(f"{datapath}/{var}/{var}_case{case}.npy"))
                    except FileNotFoundError:
                        values_all[var] = np.zeros((1, 2, 1))
                        print(f"\n Could not find {var}_case {case}. Skipping...")
                        break
                    except:
                        print(f"\n Something wrong when loading {var} case {case}.")
                        exit()
                    values_all[var] += [values]
                values_all[var] = np.asarray(values_all[var])
            # Calculate metric
            values, labels = func(*values_all.values(), )
            for value, label in zip(values, labels):
                export_dict[f"{metric_name}{label}"] = value.tolist()
        # Export metrics to json file
        with open(os.path.abspath(f"{run_path}/tests/{test}/metrics.json"), 'w') as f:
            json.dump(export_dict, f, indent="    ")
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate metrics from test simulations')
    parser.add_argument("run_path", help="Paths to folders containing model data")
    parser.add_argument("--metrics", help="Which metrics will be computed", nargs='+', default=None)
    parser.add_argument("--tests", help="Tests that will be used to calculate metrics", nargs='+', default=None)
    args = parser.parse_args()
    runs_path = args.run_path
    metrics = args.metrics
    tests = args.tests
    execute(runs_path, rotation_metrics=False, metrics_keys=metrics, tests=tests)

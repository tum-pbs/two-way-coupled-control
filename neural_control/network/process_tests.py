import collections
import numpy as np
import os
from natsort import natsorted
import matplotlib.pyplot as plt
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

if __name__ == '__main__':
    fig, axes = plt.subplots(1, 3, figsize=[20, 10])
    # Loop through models test folders
    models = {
        "1": {
            "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/test_unsupervised_model49_2/data/",
            "id": "no_integrator",
            "error_x": [],
            "error_y": [],
            "error_ang": [],
            "color": colors[0]},
        "2": {
            "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/test_unsupervised_model49_integrator/data/",
            "id": "with_integrator",
            "error_x": [],
            "error_y": [],
            "error_ang": [],
            "color": colors[1]},
    }
    for attrs in models.values():
        all_files = os.listdir(f"{attrs['path']}/error_ang/")
        all_files = [file.split("error_ang")[1] for file in all_files]  # Remove prefix
        cases = natsorted(tuple(set([file.split("case")[1][:4] for file in all_files])))
        snapshots = natsorted(tuple(set((file.split("_")[-1][:4] for file in all_files))))
        all_error_x, all_error_y, all_error_ang = [], [], []
        # Loop through cases
        for case in cases:
            # Load angular and spatial error of all snapshots
            error_ang = np.array([np.load(f"{attrs['path']}/error_ang/error_ang_case{case}_{snapshot}.npy") for snapshot in snapshots])
            error_x = np.array([np.load(f"{attrs['path']}/error_x/error_x_case{case}_{snapshot}.npy") for snapshot in snapshots])
            error_y = np.array([np.load(f"{attrs['path']}/error_y/error_y_case{case}_{snapshot}.npy") for snapshot in snapshots])
            # Normalize by initial value
            error_ang /= error_ang[0]
            error_x /= error_x[0]
            error_y /= error_y[0]
            # Store it
            all_error_x += [error_x]
            all_error_y += [error_y]
            all_error_ang += [error_ang]
        # Store as a numpy matrix
        all_error_x = np.array(all_error_x)
        all_error_y = np.array(all_error_y)
        all_error_ang = np.array(all_error_ang)
        # Take mean and standard deviation accross all cases for each snapshot
        mean_error_x = np.mean(all_error_x, 0)
        sigma_error_x = np.std(all_error_x, 0)
        mean_error_y = np.mean(all_error_y, 0)
        sigma_error_y = np.std(all_error_y, 0)
        mean_error_ang = np.mean(all_error_ang, 0)
        sigma_error_ang = np.std(all_error_ang, 0)

        axes[0].plot(mean_error_x, color=attrs["color"], label=attrs["id"])
        # axes[0].plot(mean_error_x + sigma_error_x, linestyle='--', color=attrs["color"])
        # axes[0].plot(mean_error_x - sigma_error_x, linestyle='--', color=attrs["color"])
        for line in all_error_x: axes[0].plot(line, color=attrs["color"], alpha=0.2)
        axes[1].plot(mean_error_y, color=attrs["color"], label=attrs["id"])
        # axes[1].plot(mean_error_y + sigma_error_y, linestyle='--', color=attrs["color"])
        # axes[1].plot(mean_error_y - sigma_error_y, linestyle='--', color=attrs["color"])
        for line in all_error_y: axes[1].plot(line, color=attrs["color"], alpha=0.2)
        axes[2].plot(mean_error_ang, color=attrs["color"], label=attrs["id"])
        # axes[2].plot(mean_error_ang + sigma_error_ang, linestyle='--', color=attrs["color"])
        # axes[2].plot(mean_error_ang - sigma_error_ang, linestyle='--', color=attrs["color"])
        for line in all_error_ang: axes[2].plot(line, color=attrs["color"], alpha=0.2)

    for ax in axes:
        ax.grid(True, linestyle="--")
        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        plt.legend(handle_list, label_list)
    axes[0].set_title("error_x")
    axes[1].set_title("error_y")
    axes[2].set_title("error_ang")

    plt.show(block=True)

    # Store it

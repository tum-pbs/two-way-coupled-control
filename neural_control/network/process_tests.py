import collections
from itertools import cycle
import numpy as np
import os
from natsort import natsorted
import matplotlib.pyplot as plt
colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
hatchs = cycle(['/', '\\', '|', '-', '.', '*', 'o', 'O'])

if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2, figsize=[20, 10])
    # Loop through models test folders
    models = {
        # "model0099":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/local_model0099/data/",
        #     "id": "local_0099",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs),
        # },
        # "model0099_int":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/local_model0099_integrator/data/",
        #     "id": "local_0099_int",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs),
        # },
        # "model0199":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/local_model0199/data/",
        #     "id": "local_0199",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs),
        # },
        # "model0199_int":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/local_model0199_integrator/data/",
        #     "id": "local_0199_integrator",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs),
        # },
        # "model0199_int2":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/local_model0199_integrator_v2/data/",
        #     "id": "local_0199_integrator_v2",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs),
        # },
        # "model0050":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/local_model0050/data/",
        #     "id": "local_0050",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs),
        # },
        # "3": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/PID/data/",
        #     "id": "PID",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)
        # },
        # "angle_loss": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/tests/more_angle_midtraining/data/",
        #     "id": "loss_angle_midtraining",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)
        # },
        # "all_midtraining": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/tests/loss_allterms_midtraining/data/",
        #     "id": "loss_allterms_midtraining",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # "all": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/tests/loss_allterms/data/",
        #     "id": "loss_allterms",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # "all_decay": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/tests/loss_allterms_lrdecay/data/",
        #     "id": "loss_allterms_decay",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # ***************************************************************************************
        # ************************************** Translation ************************************
        # ***************************************************************************************
        # "pw00": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/fc_pw00/data/",
        #     "id": "pw00",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # "pw01": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/fc_pw01/data/",
        #     "id": "pw01",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # "pw02": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/fc_pw02/data/",
        #     "id": "pw02",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # "pw03": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/fc_pw03/data/",
        #     "id": "pw03",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # "lstm01": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/lstm_pw01/data/",
        #     "id": "lstm01",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        #     "lstm02": {
        #         "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/lstm_pw02/data/",
        #         "id": "lstm02",
        #         "error_x": [],
        #         "error_y": [],
        #         "error_ang": [],
        #         "color": next(colors),
        #         "hatch": next(hatchs)},
        #     "lstm03": {
        #         "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/lstm_pw03/data/",
        #         "id": "lstm03",
        #         "error_x": [],
        #         "error_y": [],
        #         "error_ang": [],
        #         "color": next(colors),
        #         "hatch": next(hatchs)},
        # "unsupervised": {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/unsupervised/data/",
        #     "id": "unsupervised",
        #     "error_x": [],
        #     "error_y": [],
        #     "error_ang": [],
        #     "color": next(colors),
        #     "hatch": next(hatchs)},
        # ************************Stronger inflow************************
        "unsupervised": {
            "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/unsupervised_inflow_doubled/data/",
            "id": "unsupervised",
            "error_x": [],
            "error_y": [],
            "error_ang": [],
            "color": next(colors),
            "hatch": next(hatchs)},
        "fc_pw00": {
            "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/fc_pw00_inflow_doubled/data/",
            "id": "fc_pw00",
            "error_x": [],
            "error_y": [],
            "error_ang": [],
            "color": next(colors),
            "hatch": next(hatchs)},

    }
    for attrs in models.values():
        all_files = os.listdir(f"{attrs['path']}/error_x/")
        all_files = [file.split("error_x")[1] for file in all_files]  # Remove prefix
        cases = natsorted(tuple(set([file.split("case")[1][:4] for file in all_files])))
        snapshots = natsorted(tuple(set((file.split("_")[-1][:4] for file in all_files))))
        all_error_x, all_error_y, all_error_ang, all_error = [], [], [], []
        # Loop through cases
        for case in cases:
            # Load angular and spatial error of all snapshots
            error_x = np.array([np.load(f"{attrs['path']}/error_x/error_x_case{case}_{snapshot}.npy") for snapshot in snapshots])
            error_y = np.array([np.load(f"{attrs['path']}/error_y/error_y_case{case}_{snapshot}.npy") for snapshot in snapshots])
            try:
                error_ang = np.array([np.load(f"{attrs['path']}/error_ang/error_ang_case{case}_{snapshot}.npy") for snapshot in snapshots])
            except:
                print("No angle data")
                error_ang = np.array((np.zeros_like(error_x,)))
            # Normalize by initial value
            error_ang /= np.pi  # error_ang[0]
            error = np.sqrt(error_x**2 + error_y**2)
            error /= 20.  # error[0]
            error_x /= 20.  # error_x[0]
            error_y /= 20.  # error_y[0]
            # Store it
            all_error_x += [error_x]
            all_error_y += [error_y]
            all_error_ang += [error_ang]
            all_error += [error]
        # Store as a numpy matrix
        all_error_x = np.array(all_error_x)
        all_error_y = np.array(all_error_y)
        all_error_ang = np.array(all_error_ang)
        all_error = np.array(all_error)
        # Take mean and standard deviation accross all cases for each snapshot
        mean_error_x = np.mean(all_error_x, 0)
        sigma_error_x = np.std(all_error_x, 0)
        mean_error_y = np.mean(all_error_y, 0)
        sigma_error_y = np.std(all_error_y, 0)
        mean_error_ang = np.mean(all_error_ang, 0)
        sigma_error_ang = np.std(all_error_ang, 0)
        mean_error = np.mean(all_error, 0)
        sigma_error = np.std(all_error, 0)

        alpha = 0.15
        # axes[0].plot(mean_error_x, color=attrs["color"], label=attrs["id"])
        # # axes[0].plot(mean_error_x + sigma_error_x, linestyle='--', color=attrs["color"])
        # # axes[0].plot(mean_error_x - sigma_error_x, linestyle='--', color=attrs["color"])
        # for line in all_error_x: axes[0].plot(line, color=attrs["color"], alpha=alpha)
        # axes[1].plot(mean_error_y, color=attrs["color"], label=attrs["id"])
        # # axes[1].plot(mean_error_y + sigma_error_y, linestyle='--', color=attrs["color"])
        # # axes[1].plot(mean_error_y - sigma_error_y, linestyle='--', color=attrs["color"])
        # for line in all_error_y: axes[1].plot(line, color=attrs["color"], alpha=alpha)
        x = np.arange(np.size(mean_error_ang))
        axes[1].plot(mean_error_ang, color=attrs["color"], label=attrs["id"])
        axes[1].fill_between(x, mean_error_ang - sigma_error_ang, mean_error_ang + sigma_error_ang, hatch=attrs["hatch"], alpha=0.2, edgecolor=attrs["color"])
        # axes[1].plot(mean_error_ang + sigma_error_ang, linestyle='--', color=attrs["color"])
        # axes[1].plot(mean_error_ang - sigma_error_ang, linestyle='--', color=attrs["color"])
        # for line in all_error_ang: axes[1].plot(line, color=attrs["color"], alpha=alpha)

        axes[0].plot(mean_error, color=attrs["color"], label=attrs["id"])
        # axes[0].plot(mean_error, color=attrs["color"], label=attrs["id"])
        axes[0].fill_between(x, mean_error - sigma_error, mean_error + sigma_error, hatch=attrs["hatch"], alpha=0.2, edgecolor=attrs["color"])
        # for line in all_error: axes[0].plot(line, color=attrs["color"], alpha=alpha)

    for ax in axes:
        # ax.grid(True, linestyle="--")
        ax.plot([0, np.size(mean_error) - 1], [0, 0], 'k--')
        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        plt.legend(handle_list, label_list)
    # axes[0].set_title("error_x")
    # axes[1].set_title("error_y")
    axes[0].set_title("error_xy")
    axes[1].set_title("error_ang")

    plt.show(block=True)

    # Store it

import argparse
from natsort import natsorted
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# %% Plot parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'

# Default colors used in the paper
"""
Diff - b
PID - y
Sup - g
LS - r
RL - p
"""

# Figure parameters
plt.rc('axes', axisbelow=True)
plt.rcParams.update({'font.size': 10})
x_scale_factor = {
    1: 0.1,
    2: 0.05,
    3: 0.1,
    4: 0.05,
    5: 0.05,
    6: 0.05,
    7: 0.05,
    8: 0.05,
    9: 0.05,
    10: 0.05,
}
height, width = [1.8, 4.0]
colors_hash = dict(
    b='#1f77b4',
    y='#ff7f0e',
    g='#2ca02c',
    r='#d62728',
    p='#9467bd',
)
fig_xy, ax_xy = plt.subplots(1, 1, figsize=(width, height))
plt.tight_layout(w_pad=60)
fig_ang, ax_ang = plt.subplots(1, 1, figsize=(width, height))
plt.tight_layout(w_pad=100)
ax_xy.grid(True, linestyle='--')
ax_ang.grid(True, linestyle='--')
ax_xy.set_ylabel('$e_{xy}$')
ax_ang.set_ylabel('${e_{\\alpha}}$')
ax_xy.set_xlabel("$t$")
ax_ang.set_xlabel("$t$")
# %% Inputs
parser = argparse.ArgumentParser(description='Plot steady state errors')
# Path to root folder
parser.add_argument("root", help="Path to root folder")
# Test ID
parser.add_argument("test", help="Test ID", type=int)
# Models folders
parser.add_argument("--folders", "-f", nargs='+', help="Models folders")
# Colors
parser.add_argument("--colors", "-c", nargs='+', help="Lines colors")
# Labels
parser.add_argument("--labels", "-l", nargs='+', help="Folders labels")

args = parser.parse_args()
root = args.root
test = args.test
folders = args.folders
colors = dict(zip(folders, args.colors))
labels = dict(zip(folders, args.labels))

# %% Final plot parameters
# Find out how many simulations are there in this test
test_folder = natsorted([folder for folder in os.listdir(f"{root}/{folders[0]}/tests") if folder.split("_")[0] == f"test{test}"])[-1]
with open(os.path.abspath(f"{root}/{folders[0]}/tests/{test_folder}/tests.json"), "r") as f:
    data = json.load(f)
n_simulations = len(data[f'test{test}']) - 2
os.makedirs(os.path.join(root, 'figs'), exist_ok=True)
# %% Plot lines
for i in range(n_simulations):
    # Remove lines of plot
    ax_xy.lines.clear()
    ax_ang.lines.clear()
    for folder in folders:
        test_folder = natsorted([folder for folder in os.listdir(f"{root}/{folder}/tests") if folder.split("_")[0] == f"test{test}"])[-1]
        # test_folder = f"test{test}_{model}"
        print(f"Loading data from {test_folder} sim {i} {folder} ")
        try:
            with open(os.path.abspath(f"{root}/{folder}/tests/{test_folder}/metrics.json"), "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\n Could not find metrics.json for {test_folder} in {folder} \n")
            continue
        #  Spatial error
        error_xy = np.array(data['error_xy_all'])[i]
        x = np.arange(len(error_xy)) * x_scale_factor[test]
        ax_xy.plot(x, error_xy, color=colors_hash[colors[folder]], alpha=0.5, label=labels[folder])
        ax_xy.set_ylim([-1, error_xy.max() + 1])
        # Mean Ang error
        if len(data['error_ang_all']) != len(data['error_xy_all']): continue
        error_ang = np.array(data['error_ang_all'])[i]
        x = np.arange(len(error_ang)) * x_scale_factor[test]
        ax_ang.plot(x, error_ang, color=colors_hash[colors[folder]], alpha=0.5, label=labels[folder])
        ax_ang.set_ylim([-0.5, error_ang.max() + 0.5])

    ax_xy.legend()
    fig_xy.savefig(os.path.abspath(f"{root}/error_xy_test{test}_{i}.pdf"), bbox_inches='tight')
    if len(data['error_ang_all']) == len(data['error_xy_all']):
        # ax_ang.legend()
        fig_ang.savefig(os.path.abspath(f"{root}/error_ang_test{test}_{i}.pdf"), bbox_inches='tight')

# Mean errors
ax_xy.set_ylabel('$\|e_{xy}\|$')
ax_ang.set_ylabel('${\|e_{\\alpha}\|}$')
ax_xy.lines.clear()
ax_ang.lines.clear()
for folder in folders:
    test_folder = natsorted([folder for folder in os.listdir(f"{root}/{folder}/tests") if folder.split("_")[0] == f"test{test}"])[-1]
    # test_folder = f"test{test}_{model}"
    print(f"Loading data from {test_folder} sim {i}")
    try:
        with open(os.path.abspath(f"{root}/{folder}/tests/{test_folder}/metrics.json"), "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n Could not find metrics.json for {test_folder} in {folder} \n")
        continue
    #  Mean spatial error
    error_xy = np.array(data['error_xy'])
    error_xy_stdd = np.array(data['error_xy_stdd'])
    x = np.arange(len(error_xy)) * x_scale_factor[test]
    ax_xy.plot(x, error_xy, color=colors_hash[colors[folder]], alpha=0.5)
    ax_xy.set_ylim([-1, error_xy.max() + 1])
    # Mean Ang error
    error_ang_stdd = np.array(data['error_ang_stdd'])
    error_ang = np.array(data['error_ang'])
    x = np.arange(len(error_ang)) * x_scale_factor[test]
    ax_ang.plot(x, error_ang, color=colors_hash[colors[folder]], alpha=0.5)
    ax_ang.set_ylim([-0.5, error_ang.max() + 0.5])


fig_xy.savefig(os.path.abspath(f"{root}/figs/error_xy_test{test}_mean.pdf"), bbox_inches='tight')
if len(data['error_ang_all']) == len(data['error_xy_all']):
    fig_ang.savefig(os.path.abspath(f"{root}/figs/error_ang_test{test}_mean.pdf"), bbox_inches='tight')

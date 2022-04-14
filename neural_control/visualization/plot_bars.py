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

test_labels = {
    1: '\\textit{Ba2}',
    2: '\\textit{Buo2}',
    3: '\\textit{Ba3}',
    4: '\\textit{In3}',
    5: '\\textit{InBuo3}',
    6: '\\textit{InBuoAg3}',
    7: '\\textit{In3}',
    8: '\\textit{InBuo3}',
    9: '\\textit{InBuoAg3}',
    10: '\\textit{Buo2}',
}

height, width = [1.8, 1.3]
colors_hash = dict(
    b='#1f77b4',
    y='#ff7f0e',
    g='#2ca02c',
    r='#d62728',
    p='#9467bd',
)
x = np.arange(0, 1) * 0.5
max_ang = 0
max_xy = 0
fig_xy, ax_xy = plt.subplots(1, 1, figsize=(width, height))
plt.tight_layout(w_pad=60)
fig_ang, ax_ang = plt.subplots(1, 1, figsize=(width, height))
plt.tight_layout(w_pad=100)
ax_xy.grid(True, linestyle='--')
ax_ang.grid(True, linestyle='--')
ax_xy.set_ylabel('${ \\overline{\|e_{xy}\|}}_{ss}$')
ax_ang.set_ylabel('${\\overline{\|e_{\\alpha}\|}}_{ss}$')
plt.rc('axes', axisbelow=True)

# %% Inputs
parser = argparse.ArgumentParser(description='Plot steady state errors')
# Path to root folder
parser.add_argument("root", help="Path to root folder")
# Test ID
parser.add_argument("test", help="Test ID", type=int)
# Models folders
parser.add_argument("--folders", "-f", nargs='+', help="Models folders")
# Colors
parser.add_argument("--colors", "-c", nargs='+', help="Color of bar plots")
args = parser.parse_args()
root = args.root
test = args.test
folders = args.folders[::-1]
colors = dict(zip(folders, args.colors[::-1]))
# %% Final plot parameters
offsets = np.linspace(0, 1, len(folders))
offsets -= 0.5
offsets *= 0.25
bar_width = offsets[1] - offsets[0]
offsets = iter(-offsets)
# %% Loop through folders and create bar plot
os.makedirs(os.path.join(root, 'figs'), exist_ok=True)
for folder in folders:
    test_folder = natsorted([folder for folder in os.listdir(f"{root}/{folder}/tests") if folder.split("_")[0] == f"test{test}"])[-1]
    # test_folder = f"test{test}_{model}"
    print(f"Loading data from {test_folder}")
    try:
        with open(os.path.abspath(f"{root}/{folder}/tests/{test_folder}/metrics.json"), "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n Could not find metrics.json for {test_folder} in {folder} \n")
        continue
    #  Spatial error
    ss_error_xy = np.array(data['ss_error_xy'])
    ss_error_xy_stdd = np.array(data['ss_error_xy_stdd'])
    print(f"Mean error xy of {folder} test {test}: {ss_error_xy} ± {ss_error_xy_stdd}")
    offset = next(offsets)
    ax_xy.bar(offset, ss_error_xy, width=bar_width, color=colors_hash[colors[folder]], alpha=0.8)
    # Draw error bars with horizontal arrows at ends
    ax_xy.errorbar(offset, ss_error_xy, yerr=ss_error_xy_stdd, color='black', linewidth=1, capsize=3, capthick=1, linestyle='-')
    max_xy = np.max((ss_error_xy + ss_error_xy_stdd, max_xy))
    ax_xy.set_ylim([0, max_xy * 1.05])
    # Ang error
    ss_error_ang_stdd = np.array(data['ss_error_ang_stdd'])
    ss_error_ang = np.array(data['ss_error_ang'])
    ax_ang.bar(offset, ss_error_ang, width=bar_width, color=colors_hash[colors[folder]], alpha=0.8)
    ax_ang.errorbar(offset, ss_error_ang, yerr=ss_error_ang_stdd, color='black', linewidth=1, capsize=3, capthick=1, linestyle='-')
    max_ang = np.max((ss_error_ang + ss_error_ang_stdd, max_ang))
    ax_ang.set_ylim([0, max_ang * 1.05])
    print(f"Mean error ang of {folder} test {test}: {ss_error_ang} ± {ss_error_ang_stdd} \n")
ax_xy.set_xticks(x)
ax_xy.set_xticklabels([test_labels[test]])
ax_ang.set_xticks(x)
ax_ang.set_xticklabels([test_labels[test]])
# Save figs
fig_xy.savefig(os.path.abspath(f"{root}/figs/ss_error_xy_test{test}.pdf"), bbox_inches='tight')
fig_ang.savefig(os.path.abspath(f"{root}/figs/ss_error_ang_test{test}.pdf"), bbox_inches='tight')

import argparse
from collections import OrderedDict, defaultdict
from itertools import cycle
import mimic_alpha as ma
import json
import os
import shutil
from InputsManager import InputsManager
from natsort.natsort import natsorted
import numpy as np
# import plotly.express as px
# from PIL import ImageColor
# import plotly.graph_objects as go
# import plotly.io as pio
from Plotter import Plotter
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.font_manager as font_manager


plt.rcParams['font.family'] = 'Times New Roman'

root = "/home/ramos/phiflow/storage/rotation/"
folders = [
    'pid',
    # 'online_16',
]
tests = ["test6"]
field_name = "vorticity"
obs_width = 20
obs_height = 6
imshow_kwargs = dict(
    vmin=-1.5,
    vmax=1.5,
    cmap='bwr',
    interpolation='bilinear'
)
quiver_kwargs = dict(
    scale=200)
xy_kwargs = dict(
    linewidth=0.5,
    create_legend=False,
    color='k'
)
torque_offset = 8
torque_kwargs = dict(
    scale=300,
    facecolor='none',
    linewidth=1,
    width=0.0001,
    headwidth=0,
    headlength=0,
)
patches_kwargs = dict(
    edgecolor='k',
    linewidth=1,
    alpha=0.8,
    fill=False,
    linestyle='--',
)
objectives_kwargs = dict(
    markersize=4,
    marker="+",
    color='k',
    linestyle="None"
)
with open(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/figs.json'), 'r') as f:
    figs_attrs = json.load(f)
global_attrs = figs_attrs['global']
export_path_root = global_attrs['export_path']
for folder in folders:
    # Get test folder
    tests_local = os.listdir(f"{root}/{folder}/tests/")
    for test_ in tests:
        test = [test_local for test_local in tests_local if test_ in test_local][0]  # Load only test with last model
        # Get snapshots and cases
        files = os.listdir(f"{root}/{folder}/tests/{test}/data/{field_name}")
        cases = natsorted(list(set([file.split("case")[1].split("_")[0] for file in files])))
        snapshots = natsorted(list(set([file.split("_")[-1].split(".")[0] for file in files])))
        export_path = f"{export_path_root}/{folder}/{test}/{field_name}"
        shutil.rmtree(export_path, ignore_errors=True)  # Delete previous
        os.makedirs(export_path, exist_ok=True)
        # Create export folder
        for case in cases:
            # Load trajectory
            xy = np.load(f"{root}/{folder}/tests/{test}/data/obs_xy/obs_xy_case{case}.npy").transpose()
            # Load forces
            control_fx = np.load(f"{root}/{folder}/tests/{test}/data/control_force_x/control_force_x_case{case}.npy")
            control_fy = np.load(f"{root}/{folder}/tests/{test}/data/control_force_y/control_force_y_case{case}.npy")
            fluid_fx = np.load(f"{root}/{folder}/tests/{test}/data/fluid_force_x/fluid_force_x_case{case}.npy")
            fluid_fy = np.load(f"{root}/{folder}/tests/{test}/data/fluid_force_y/fluid_force_y_case{case}.npy")
            # Load _torque
            control_torque = np.load(f"{root}/{folder}/tests/{test}/data/control_torque/control_torque_case{case}.npy")
            fluid_torque = np.load(f"{root}/{folder}/tests/{test}/data/fluid_torque/fluid_torque_case{case}.npy")
            # Load angle
            angle = np.load(f"{root}/{folder}/tests/{test}/data/obs_ang/obs_ang_case{case}.npy")
            # Load objectives
            obj_x = np.load(f"{root}/{folder}/tests/{test}/data/reference_x/reference_x_case{case}.npy")
            obj_y = np.load(f"{root}/{folder}/tests/{test}/data/reference_y/reference_y_case{case}.npy")
            obj_angle = np.load(f"{root}/{folder}/tests/{test}/data/reference_ang/reference_ang_case{case}.npy")
            for i, snapshot in enumerate(snapshots):
                j = int(snapshot)
                # Create object only on first time the just update data
                # Load and add obs mask
                mask = np.load(f"{root}/{folder}/tests/{test}/data/obs_mask/obs_mask_case{case}_{snapshot}.npy").transpose() * 0.5
                # Load and add field
                field = np.load(f"{root}/{folder}/tests/{test}/data/{field_name}/{field_name}_case{case}_{snapshot}.npy").transpose()
                if i == 0:
                    p = Plotter((global_attrs['width'], global_attrs['height']))
                    p.add_data([mask, field], ['mask', 'field'])
                    # Add trajectory and current position
                    p.add_data([xy[j], xy], ['xy_now', 'xy'])
                    # Plot imshow
                    field_image, ax, fig, _ = p.imshow(['field'], 'fig', create_cbar=False, create_title=False, origin='lower', **imshow_kwargs)
                    # Hide ticks and its labels
                    ax.tick_params(axis='both', which='both', length=0)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    # Make room for colorbar
                    fig.subplots_adjust(right=0.8)
                    # Add color bar
                    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().y1 - ax.get_position().y0])
                    cbar = fig.colorbar(field_image, cax=cax, ticks=[imshow_kwargs['vmin'], 0, imshow_kwargs['vmax']])
                    # Now add transparency and mask
                    field_image.set_alpha(1 - mask)
                    mask_image, *_ = p.imshow(['mask'], 'fig', cmap='Greys', alpha=mask, create_title=False, create_cbar=False, origin='lower')
                    # Create trajectory
                    p.plot(['xy'], 'fig', **xy_kwargs)
                    # Create force arrows
                    control_quiver = ax.quiver(xy[0][j], xy[1][j], control_fx[j], control_fy[j], color="#24ED0B", **quiver_kwargs)
                    fluid_quiver = ax.quiver(xy[0][j], xy[1][j], fluid_fx[j], fluid_fy[j], color="#ECE921", **quiver_kwargs)
                    # Create _torques
                    anchor_x = np.array([np.cos(-angle[j]) * torque_offset, -np.cos(-angle[j]) * torque_offset])
                    anchor_y = np.array([np.sin(-angle[j]) * torque_offset, -np.sin(-angle[j]) * torque_offset])
                    anchor_x += xy[0][j]
                    anchor_y += xy[1][j]
                    data_v = [np.cos(angle[j]) * control_torque[j], -np.cos(angle[j]) * control_torque[j]]
                    data_u = [np.sin(angle[j]) * control_torque[j], -np.sin(angle[j]) * control_torque[j]]
                    control_torque_quiver = ax.quiver(anchor_x, anchor_y, data_u, data_v, edgecolor="#24ED0B", **torque_kwargs)
                    data_v = [np.cos(angle[j]) * fluid_torque[j], -np.cos(angle[j]) * fluid_torque[j]]
                    data_u = [np.sin(angle[j]) * fluid_torque[j], -np.sin(angle[j]) * fluid_torque[j]]
                    fluid_torque_quiver = ax.quiver(anchor_x, anchor_y, data_u, data_v, edgecolor="#ECE921", **torque_kwargs)
                    # Add rectangle to show objective
                    correction_x = - np.cos(-obj_angle[j]) * obs_width / 2 + np.sin(-obj_angle[j]) * obs_height / 2
                    correction_y = - np.sin(-obj_angle[j]) * obs_width / 2 - np.cos(-obj_angle[j]) * obs_height / 2
                    corner = (obj_x[j] + correction_x, obj_y[j] + correction_y)
                    patch = ax.add_patch(patches.Rectangle(corner, obs_width, obs_height, angle=-obj_angle[j] / np.pi * 180, **patches_kwargs))
                    # Also add crosses for the objectives points
                    ax.plot(obj_x, obj_y, **objectives_kwargs)
                else:
                    # Mask
                    mask_image.set_data(mask)
                    mask_image.set_alpha(mask)
                    # Field
                    field_image.set_data(field)
                    field_image.set_alpha((1 - mask))
                    # Quivers forces
                    control_quiver.set_UVC(control_fx[j], control_fy[j])
                    control_quiver.set_offsets([xy[0][j], xy[1][j]])
                    fluid_quiver.set_UVC(fluid_fx[j], fluid_fy[j])
                    fluid_quiver.set_offsets([xy[0][j], xy[1][j]])
                    # Quivers torque
                    anchor_x = np.array([np.cos(-angle[j]) * torque_offset, -np.cos(-angle[j]) * torque_offset])
                    anchor_y = np.array([np.sin(-angle[j]) * torque_offset, -np.sin(-angle[j]) * torque_offset])
                    anchor_x += xy[0][j]
                    anchor_y += xy[1][j]
                    new_xy = ((anchor_x[0, 0], anchor_y[0, 0]), (anchor_x[1, 0], anchor_y[1, 0]))
                    data_v = [np.cos(angle[j]) * control_torque[j], -np.cos(angle[j]) * control_torque[j]]
                    data_u = [np.sin(angle[j]) * control_torque[j], -np.sin(angle[j]) * control_torque[j]]
                    control_torque_quiver.set_UVC(data_u, data_v)
                    control_torque_quiver.set_offsets(new_xy)
                    data_v = [np.cos(angle[j]) * fluid_torque[j], -np.cos(angle[j]) * fluid_torque[j]]
                    data_u = [np.sin(angle[j]) * fluid_torque[j], -np.sin(angle[j]) * fluid_torque[j]]
                    fluid_torque_quiver.set_UVC(data_u, data_v)
                    fluid_torque_quiver.set_offsets(new_xy)
                    # Patch
                    patch.remove()
                    correction_x = - np.cos(-obj_angle[j]) * obs_width / 2 + np.sin(-obj_angle[j]) * obs_height / 2
                    correction_y = - np.sin(-obj_angle[j]) * obs_width / 2 - np.cos(-obj_angle[j]) * obs_height / 2
                    corner = (obj_x[j] + correction_x, obj_y[j] + correction_y)
                    patch = ax.add_patch(patches.Rectangle(corner, obs_width, obs_height, angle=-obj_angle[j] / np.pi * 180, **patches_kwargs))
                # Save figure
                fig.savefig(f"{export_path}/{field_name}_case{case}_{i:05d}.png", dpi=global_attrs['dpi'])

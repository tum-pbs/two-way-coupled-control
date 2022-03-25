import argparse
import matplotlib
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


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})

# root = "/home/ramos/phiflow/storage/rotation/"
root = "/home/ramos/phiflow/storage/translation/"
folders = [
    'rl',
    'loop_shaped',
    'online_11',
    'supervised',
    # 'online_20_seed100',
    'pid',
    # 'online_20_onlySE_seed1000',
    # 'online_20_onlySV_seed1000',
    # 'online_20_seed1000',
    # 'dataset_box_local'
]
snapshots = [
    # '03900'  # Test 10
    # '01900',  # Test 8
    # '03900',
    # '05900',  # Test 7
    # '07900'
    # '00000',
    # '00000',
    # '00050',
    # '00100',
    # '00150',
    # '00200',
    # '00250',
    # '00300'
    # '01900',
    # '03900',
    # '05900',
    # '07900'
    # '02300'
]
cases = [
    # 15  # Test 1 and 3
    # 0  # Test 10
    # 1  # Test 7
    # 0  # Test 8
    # 15,  # Test 9
    0, 1, 2

]
plot_elements = [
    'forces',
    'objective',
    'trajectory',
    # 'torques'
]
tests = ["test1"]
field_name = dict(
    test1='vorticity',
    test2='smoke',
    test3='vorticity',
    # test3='obs_mask',
    test4='smoke',
    test5='vorticity',
    test6='vorticity',
    test7='vorticity',
    test8='smoke',
    test9='vorticity',
    test10='smoke'
)
field_name = field_name[tests[0]]
# field_name = "vx"
# field_name = "smoke"
# obs_width = 20
obs_width = 5
cbar_labels = defaultdict(lambda: "0")
cbar_labels['vx'] = '$u_x$'
x_lim = dict(
    # test1=[30, 65],
    # test1=[35, 65],  # Test 1 snapshots
    test1=[10, 80],  # video
    # test2=[20, 60],
    test2=[10, 70],  # Video
    # test3=[25, 75], # Snapshots
    test3=[10, 70],  # Video
    # test3=[25, 85],  # Fig Schmeatic
    test4=[65, 130],
    test5=[65, 135],
    test6=[65, 135],
    # test7=[65, 130],
    # test8=[65, 135],
    test9=[60, 145],  # Video
    # test9=[65, 135],
    test7=[65, 135],  # Snapshots
    test8=[65, 135],  # Snapshots
    # test9=[65, 135],  # Snapshots
    test10=[20, 60],
    # test1=[0, 80],
    # test2=[0, 80],
    # test3=[0, 80],
    # test4=[0, 175],
    # test5=[0, 175],
    # test6=[0, 175],
    # test1=[30, 65],
    # test2=[20, 60],
)
y_lim = dict(
    # test1=[30, 70],
    # test1=[15, 45],  # snapshots
    test1=[10, 70],  # video
    # test2=[20, 72],
    test2=[15, 80],  # Video
    # test3=[25, 70],  # Snapshots
    test3=[15, 75],  # Video
    # test3=[30, 70],  # Schematic
    test4=[28, 75],
    test5=[28, 85],
    test6=[28, 80],
    test9=[28, 95],  # Videos
    # test7=[28, 75],
    # test8=[28, 85],
    # test9=[28, 85],
    test7=[25, 85],  # Snapshots
    test8=[25, 85],  # Snapshots
    # test9=[28, 85],  # Snapshots
    test10=[20, 72],
    # test1=[0, 80],
    # test2=[0, 80],
    # test3=[0, 80],
    # test4=[0, 110],
    # test5=[0, 110],
    # test6=[0, 110],
)
v = dict(
    test10=[0, 1.5],
    test7=[-2, 2],
    test8=[0, 1.5],
    test9=[-3, 3],
    test1=[-1, 1],
    test3=[-1, 1],
    # test3=[0, 1],
    test2=[0, 1.5],
)
vmin, vmax = v[tests[0]]
obs_height = 6
cmap = dict(
    test1='bwr',
    test10='Greens',
    test3='bwr',
    test7='bwr',
    test8='Greens',
    test9='bwr',
    test2='Greens',
)
imshow_kwargs = dict(
    # cmap='bwr',
    # cmap='PuOr',
    cmap=cmap[tests[0]],
    vmin=vmin,
    vmax=vmax,

    interpolation='bilinear'
)
quiver_kwargs = dict(
    # scale=100,
    scale=20,
    # scale=20,
    width=0.01,
    zorder=10
)
xy_kwargs = dict(
    linewidth=.5,
    # linewidth=1.5,
    create_legend=False,
    color='k',
    linestyle='-',
    zorder=5,
)
xy_current_kwargs = dict(
    # marker='o',
    marker='None',
    color='w',
)
torque_offset = 8
torque_kwargs = dict(
    scale=150,
    facecolor='none',
    linewidth=1,
    width=0.0002,
    headwidth=0,
    headlength=0,
    # linestyle=(1, (1, 10))
)
patches_kwargs = dict(
    edgecolor='k',
    linewidth=1.5,
    alpha=0.8,
    fill=False,
    linestyle='--',
)
objectives_kwargs = dict(
    # markersize=15,
    markersize=5,
    linewidth=4,
    marker="+",
    color='k',
    alpha=0.5,
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
        print(f"\n\n Ploting {field_name} from {folder} and {test_} \n\n")
        test = natsorted([test_local for test_local in tests_local if test_local.split("_")[0] == test_])[-1]  # Load only test with last model
        # Get snapshots and cases
        files = os.listdir(f"{root}/{folder}/tests/{test}/data/{field_name}")
        if cases == []: cases = natsorted(list(set([file.split("case")[1].split("_")[0] for file in files])))
        if snapshots == []: snapshots = natsorted(list(set([file.split("_")[-1].split(".")[0] for file in files])))
        export_path = f"{export_path_root}/{folder}/{test_}/{field_name}"
        # shutil.rmtree(export_path, ignore_errors=True)  # Delete previous
        os.makedirs(export_path, exist_ok=True)
        # Create export folder
        for case in cases:
            # Load trajectory
            xy = np.load(f"{root}/{folder}/tests/{test}/data/obs_xy/obs_xy_case{case:04d}.npy").transpose() - 0.5  # Remove offset for visualization
            # Load forces
            control_fx = np.load(f"{root}/{folder}/tests/{test}/data/control_force_x/control_force_x_case{case:04d}.npy")
            control_fy = np.load(f"{root}/{folder}/tests/{test}/data/control_force_y/control_force_y_case{case:04d}.npy")
            fluid_fx = np.load(f"{root}/{folder}/tests/{test}/data/fluid_force_x/fluid_force_x_case{case:04d}.npy")
            fluid_fy = np.load(f"{root}/{folder}/tests/{test}/data/fluid_force_y/fluid_force_y_case{case:04d}.npy")
            # Load objectives
            obj_x = np.load(f"{root}/{folder}/tests/{test}/data/reference_x/reference_x_case{case:04d}.npy") - 0.5  # Remove offset for visualization
            obj_y = np.load(f"{root}/{folder}/tests/{test}/data/reference_y/reference_y_case{case:04d}.npy") - 0.5  # Remove offset for visualization
            try:
                # Load _torque
                control_torque = np.load(f"{root}/{folder}/tests/{test}/data/control_torque/control_torque_case{case:04d}.npy")
                fluid_torque = np.load(f"{root}/{folder}/tests/{test}/data/fluid_torque/fluid_torque_case{case:04d}.npy")
                # Load angle
                angle = np.load(f"{root}/{folder}/tests/{test}/data/obs_ang/obs_ang_case{case:04d}.npy")
                obj_angle = np.load(f"{root}/{folder}/tests/{test}/data/reference_ang/reference_ang_case{case:04d}.npy")
                patch_func = patches.Rectangle
            except FileNotFoundError:
                print("Treating simulation as translation only")
                control_torque = np.zeros_like(control_fx)
                fluid_torque = np.zeros_like(fluid_fx)
                angle = np.zeros_like(control_fx)
                obj_angle = np.zeros_like(control_fx)
                patch_func = patches.Circle
                pass
            for i, snapshot in enumerate(snapshots):
                j = int(snapshot)
                # Create object only on first time the just update data
                # Load and add obs mask
                mask = np.load(f"{root}/{folder}/tests/{test}/data/obs_mask/obs_mask_case{case:04d}_{snapshot}.npy").transpose() * 0.5
                # Load and add field
                field = np.load(f"{root}/{folder}/tests/{test}/data/{field_name}/{field_name}_case{case:04d}_{snapshot}.npy").transpose()
                if i == 0:
                    p = Plotter((global_attrs['width'], global_attrs['height']))
                    p.add_data([mask, field], ['mask', 'field'])
                    # Plot imshow
                    field_image, ax, fig, _ = p.imshow(['field'], 'fig', create_cbar=False, create_title=False, origin='lower', **imshow_kwargs)
                    # Hide ticks and its labels
                    ax.tick_params(axis='both', which='both', length=0)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    # Make room for colorbar
                    fig.subplots_adjust(right=0.8)
                    # Add color bar
                    cax = fig.add_axes([ax.get_position().x1 + 0.04, ax.get_position().y0, 0.02, ax.get_position().y1 - ax.get_position().y0])
                    cbar = fig.colorbar(field_image, cax=cax, ticks=[imshow_kwargs['vmin'], 0, imshow_kwargs['vmax']])
                    cbar.ax.set_yticklabels([imshow_kwargs['vmin'], cbar_labels[field_name], imshow_kwargs['vmax']])
                    # Now add transparency and mask
                    field_image.set_alpha(1 - mask)
                    mask_image, *_ = p.imshow(['mask'], 'fig', cmap='Greys', alpha=mask, create_title=False, create_cbar=False, origin='lower')
                    if 'trajectory' in plot_elements:
                        # Add trajectory and current position
                        p.add_data([xy], ['xy'])
                        # Create trajectory
                        p.plot(['xy'], 'fig', **xy_kwargs)
                        # Add current position
                        p.add_data([xy[0:2, j:j + 1]], ['xy_current'])
                        current_xy = p.plot(['xy_current'], 'fig', create_legend=False, **xy_current_kwargs)[0][0]
                    if 'objective' in plot_elements:
                        # Add rectangle to show objective
                        if patch_func == patches.Rectangle:
                            correction_x = - np.cos(-obj_angle[j]) * obs_width / 2 + np.sin(-obj_angle[j]) * obs_height / 2
                            correction_y = - np.sin(-obj_angle[j]) * obs_width / 2 - np.cos(-obj_angle[j]) * obs_height / 2
                            corner = (obj_x[j] + correction_x, obj_y[j] + correction_y)
                            patch = ax.add_patch(patch_func(corner, obs_width, obs_height, angle=-obj_angle[j] / np.pi * 180, **patches_kwargs))
                        if patch_func == patches.Circle:
                            patch = ax.add_patch(patch_func([obj_x[j], obj_y[j]], radius=obs_width, **patches_kwargs))
                        # Also add crosses for the objectives points
                        ax.plot(obj_x, obj_y, **objectives_kwargs)
                    if 'forces' in plot_elements:
                        # Create force arrows
                        control_quiver = ax.quiver(xy[0][j], xy[1][j], control_fx[j], control_fy[j], color="tab:orange", **quiver_kwargs)
                        fluid_quiver = ax.quiver(xy[0][j], xy[1][j], fluid_fx[j], fluid_fy[j], color="tab:olive", **quiver_kwargs)
                    if 'torques' in plot_elements:
                        # Create _torques
                        anchor_x = np.array([np.cos(-angle[j]) * torque_offset, -np.cos(-angle[j]) * torque_offset])
                        anchor_y = np.array([np.sin(-angle[j]) * torque_offset, -np.sin(-angle[j]) * torque_offset])
                        anchor_x += xy[0][j]
                        anchor_y += xy[1][j]
                        data_v = [np.cos(angle[j]) * control_torque[j], -np.cos(angle[j]) * control_torque[j]]
                        data_u = [np.sin(angle[j]) * control_torque[j], -np.sin(angle[j]) * control_torque[j]]
                        control_torque_quiver = ax.quiver(anchor_x, anchor_y, data_u, data_v, edgecolor="tab:orange", **torque_kwargs)
                        data_v = [np.cos(angle[j]) * fluid_torque[j], -np.cos(angle[j]) * fluid_torque[j]]
                        data_u = [np.sin(angle[j]) * fluid_torque[j], -np.sin(angle[j]) * fluid_torque[j]]
                        fluid_torque_quiver = ax.quiver(anchor_x, anchor_y, data_u, data_v, edgecolor="tab:olive", **torque_kwargs)
                else:
                    # Mask
                    mask_image.set_data(mask)
                    mask_image.set_alpha(mask)
                    # Field
                    field_image.set_data(field)
                    field_image.set_alpha((1 - mask))
                    if 'trajectory' in plot_elements:
                        # Current position
                        current_xy.set_xdata(xy[0][j])
                        current_xy.set_ydata(xy[1][j])
                    if 'forces' in plot_elements:
                        # Quivers forces
                        control_quiver.set_UVC(control_fx[j], control_fy[j])
                        control_quiver.set_offsets([xy[0][j], xy[1][j]])
                        fluid_quiver.set_UVC(fluid_fx[j], fluid_fy[j])
                        fluid_quiver.set_offsets([xy[0][j], xy[1][j]])
                    if 'torques' in plot_elements:
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
                    if 'objective' in plot_elements:
                        # Patch
                        patch.remove()
                        # Add rectangle to show objective
                        if patch_func == patches.Rectangle:
                            correction_x = - np.cos(-obj_angle[j]) * obs_width / 2 + np.sin(-obj_angle[j]) * obs_height / 2
                            correction_y = - np.sin(-obj_angle[j]) * obs_width / 2 - np.cos(-obj_angle[j]) * obs_height / 2
                            corner = (obj_x[j] + correction_x, obj_y[j] + correction_y)
                            patch = ax.add_patch(patch_func(corner, obs_width, obs_height, angle=-obj_angle[j] / np.pi * 180, **patches_kwargs))
                        if patch_func == patches.Circle:
                            patch = ax.add_patch(patch_func([obj_x[j], obj_y[j]], radius=obs_width, **patches_kwargs))
                ax.set_xlim(x_lim[test_])
                ax.set_ylim(y_lim[test_])
                # Save figure
                fig.savefig(f"{export_path}/{field_name}_case{case:04d}_{snapshot}.png", dpi=global_attrs['dpi'])
                # exit()

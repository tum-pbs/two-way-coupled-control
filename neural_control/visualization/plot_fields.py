import os
from natsort.natsort import natsorted
import numpy as np
from neural_control.visualization.Plotter import Plotter
import matplotlib.pyplot as plt
from matplotlib import patches
import argparse

# %% Plot parameters

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})
# width = 4
width = 2
height = 1.8
plot_elements = [
    # 'forces',
    'objective',
    'trajectory',
    # 'torques'
]
field_name = {
    1: 'vorticity',
    2: 'smoke',
    3: 'vorticity',
    4: 'smoke',
    5: 'vorticity',
    6: 'vorticity',
    7: 'vorticity',
    8: 'smoke',
    9: 'vorticity',
    10: 'smoke'
}
cmap = {
    1: 'bwr',
    10: 'Greens',
    3: 'bwr',
    7: 'bwr',
    8: 'Greens',
    9: 'bwr',
    2: 'Greens',
}


quiver_kwargs = dict(
    # scale=100,
    scale=20,
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
    marker='o',
    # marker='None',
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
obs_height = 6

x_lim = {
    # 1: [30, 65],
    1: [35, 65],  # Test 1 snapshots
    # 1: [10, 80],  # video
    2: [20, 60],
    # 2: [10, 70],  # Video
    3: [25, 75],  # Snapshots
    # 3: [10, 70],  # Video
    # 3: [25, 85],  # Fig Schmeatic
    4: [65, 130],
    5: [65, 135],
    6: [65, 135],
    # 7: [65, 130],
    # 8: [65, 135],
    # 9: [60, 145],  # Video
    # 9: [65, 135],
    7: [65, 135],  # Snapshots
    8: [65, 135],  # Snapshots
    9: [65, 135],  # Snapshots
    10: [20, 60],
}
y_lim = {
    # 1: [30, 70],
    1: [15, 45],  # snapshots
    # 1: [10, 70],  # video
    2: [20, 72],
    # 2: [15, 80],  # Video
    3: [25, 70],  # Snapshots
    # 3: [15, 75],  # Video
    # 3: [30, 70],  # Schematic
    4: [28, 75],
    5: [28, 85],
    6: [28, 80],
    # 9: [28, 95],  # Videos
    # 7: [28, 75],
    # 8: [28, 85],
    # 9: [28, 85],
    7: [25, 85],  # Snapshots
    8: [25, 85],  # Snapshots
    9: [28, 85],  # Snapshots
    10: [20, 72],
}
v = {
    10: [0, 1.5],
    7: [-2, 2],
    8: [0, 1.5],
    9: [-3, 3],
    1: [-1, 1],
    3: [-1, 1],
    2: [0, 1.5],
}
# %% Inputs
parser = argparse.ArgumentParser(description='Plot fields')
# Path to root folder
parser.add_argument("root", help="Path to root folder")
# Test ID
parser.add_argument("test", help="Test ID", type=int)
# Snapshots
parser.add_argument("snapshots", nargs='+', help="Snapshots to be plotted", type=int)
# Models folders
parser.add_argument("--folders", "-f", nargs='+', help="Models folders")
args = parser.parse_args()
root = args.root
folders = args.folders
snapshots = args.snapshots
test = args.test

# %% Final plotting paramters
field_name = field_name[test]
obs_width = 5 if test in [1, 2, 10] else 20
imshow_kwargs = dict(
    cmap=cmap[test],
    vmin=v[test][0],
    vmax=v[test][1],
    interpolation='bilinear'
)
# %% Loop through folders and plot fields
os.makedirs(os.path.join(root, 'figs'), exist_ok=True)
for folder in folders:
    # Get test folder
    tests_local = os.listdir(f"{root}/{folder}/tests/")
    print(f"\n\n Ploting {field_name} from {folder} and test {test} \n\n")
    test_ = natsorted([test_local for test_local in tests_local if test_local.split("_")[0] == f"test{test}"])[-1]  # Load only test with last model
    # Get snapshots and cases
    files = os.listdir(f"{root}/{folder}/tests/{test_}/data/{field_name}")
    cases = natsorted(list(set([file.split("case")[1].split("_")[0] for file in files])))
    # Create export folder
    for case in cases:
        # Load trajectory
        xy = np.load(f"{root}/{folder}/tests/{test_}/data/obs_xy/obs_xy_case{case}.npy").transpose() - 0.5  # Remove offset for visualization
        # Load forces
        control_fx = np.load(f"{root}/{folder}/tests/{test_}/data/control_force_x/control_force_x_case{case}.npy")
        control_fy = np.load(f"{root}/{folder}/tests/{test_}/data/control_force_y/control_force_y_case{case}.npy")
        fluid_fx = np.load(f"{root}/{folder}/tests/{test_}/data/fluid_force_x/fluid_force_x_case{case}.npy")
        fluid_fy = np.load(f"{root}/{folder}/tests/{test_}/data/fluid_force_y/fluid_force_y_case{case}.npy")
        # Load objectives
        obj_x = np.load(f"{root}/{folder}/tests/{test_}/data/reference_x/reference_x_case{case}.npy") - 0.5  # Remove offset for visualization
        obj_y = np.load(f"{root}/{folder}/tests/{test_}/data/reference_y/reference_y_case{case}.npy") - 0.5  # Remove offset for visualization
        try:
            # Load _torque
            control_torque = np.load(f"{root}/{folder}/tests/{test_}/data/control_torque/control_torque_case{case}.npy")
            fluid_torque = np.load(f"{root}/{folder}/tests/{test_}/data/fluid_torque/fluid_torque_case{case}.npy")
            # Load angle
            angle = np.load(f"{root}/{folder}/tests/{test_}/data/obs_ang/obs_ang_case{case}.npy")
            obj_angle = np.load(f"{root}/{folder}/tests/{test_}/data/reference_ang/reference_ang_case{case}.npy")
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
            mask = np.load(f"{root}/{folder}/tests/{test_}/data/obs_mask/obs_mask_case{case}_{snapshot:05d}.npy").transpose() * 0.5
            # Load and add field
            field = np.load(f"{root}/{folder}/tests/{test_}/data/{field_name}/{field_name}_case{case}_{snapshot:05d}.npy").transpose()
            if i == 0:
                p = Plotter((width, height))
                p.add_data([mask, field], ['mask', 'field'])
                # Plot imshow
                field_image, ax, fig, _ = p.imshow(['field'], 'fig', create_cbar=False, create_title=False, origin='lower', **imshow_kwargs)
                # Hide ticks and its labels
                ax.tick_params(axis='both', which='both', length=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # Make room for colorbar
                # fig.subplots_adjust(right=0.8)
                # Add color bar
                # cax = fig.add_axes([ax.get_position().x1 + 0.04, ax.get_position().y0, 0.02, ax.get_position().y1 - ax.get_position().y0])
                # cbar = fig.colorbar(field_image, cax=cax, ticks=[imshow_kwargs['vmin'], 0, imshow_kwargs['vmax']])
                # cbar.ax.set_yticklabels([imshow_kwargs['vmin'], cbar_labels[field_name], imshow_kwargs['vmax']])
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
            ax.set_xlim(x_lim[test])
            ax.set_ylim(y_lim[test])
            # Save figure
            fig.savefig(f"{root}/figs/{folder}_{field_name}_test{test}_case{case}_{snapshot:05d}.png", dpi=400)

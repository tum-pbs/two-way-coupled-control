import time
import argparse
import os

from InputsManager import InputsManager
from neural_control.linear_controllers.LSController import LSController
from neural_control.linear_controllers.PIDController import PIDController
from misc_funcs import *
CUDA_LAUNCH_BLOCKING = 1
torch.set_printoptions(sci_mode=True)
if __name__ == "__main__":

    # ----------------------------------------------
    # -------------------- Inputs ------------------
    # ----------------------------------------------
    parser = argparse.ArgumentParser(description='Run test simulations with obstacle controlled by model')
    parser.add_argument("export_folder", help="Path to file containing controller coefficients")
    parser.add_argument("controller_type", help="ls or pid")
    parser.add_argument("tests_id", nargs="+", help="ID of tests to be performed")
    # Add an argument to device if rotation will be turned on or not
    args = parser.parse_args()
    export_folder = args.export_folder
    tests_id = [f"test{i}" for i in args.tests_id]
    inp = InputsManager(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json"), ["translation_only", "controller"])
    inp.controller['type'] = args.controller_type
    if inp.controller["type"] == "ls":
        controller = LSController(inp.controller["ls_coeffs"])
    elif inp.controller["type"] == "pid":
        controller = PIDController(inp.controller["pid_coeffs"], 0, inp.controller["clamp_values"])
    else: raise ValueError("Invalid controller type")
    inp.controller["coefs"] = controller.get_coeffs()
    inp.export(export_folder + "/inputs.json")
    print(f"\n\n Running tests with controller {inp.controller}")
    if torch.cuda.is_available():
        TORCH_BACKEND.set_default_device("GPU")
        device = torch.device("cuda:0")
        device_name = "GPU"
    else:
        TORCH_BACKEND.set_default_device("CPU")
        device = torch.device("cpu")
        device_name = "CPU"
    tests = InputsManager(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json"), tests_id)
    for test_label, test in tests.__dict__.items():  # Loop through tests
        # ----------------------------------------------
        # ---------------- Setup simulation ------------
        # ----------------------------------------------
        inp.delete_attributes(["simulation", "export_stride"])
        inp.add_values(test["initial_conditions_path"] + "/inputs.json", ["export_stride", "simulation"])
        # Create list of scalar variables that will be exported every step
        # export_vars_scalar = list(inp.export_vars)
        export_vars = [
            "pressure",
            "vx",
            "vy",
            "obs_mask",
            "obs_xy",
            "obs_vx",
            "obs_vy",
            "control_force_x",
            "control_force_y",
            "fluid_force_x",
            "fluid_force_y",
            "reference_x",
            "reference_y",
            "error_x",
            "error_y",
            "vorticity",
            "cfl",
        ]
        if not inp.translation_only:
            export_vars += [
                "obs_ang",
                "obs_ang_vel",
                "fluid_torque",
                "error_ang",
                "control_torque",
                "reference_ang"
            ]
        export_vars_scalar = list(export_vars)
        for entry in list(export_vars_scalar):
            if entry in ["pressure", "vx", "vy", "obs_mask", "vorticity"]:
                export_vars_scalar.remove(entry)
        # Save tests used in this script
        export_path = f"{export_folder}/tests/{test_label}_/"
        print(f"\n Data will be saved on {export_path} \n")
        tests.export(export_path + "tests.json", only=[test_label, "dataset_path", "tvt_ratio"])
        sim = TwoWayCouplingSimulation(device_name, inp.translation_only)
        sim.set_initial_conditions(
            inp.simulation["obs_type"],
            inp.simulation['obs_width'],
            inp.simulation['obs_height'],
            test["initial_conditions_path"],
            obs_xy=inp.simulation['obs_xy']
        )
        # Make sure control sampling is the same as the one it was trained for
        sampling_stride = controller.dt / inp.simulation["dt"]
        assert sampling_stride.is_integer()
        last = 0
        controller.export(os.path.abspath(export_path))
        # ----------------------------------------------
        # ------------------ Simulate ------------------
        # ----------------------------------------------
        with torch.no_grad():
            is_first_export = True  # Used for deleting previous files on folder TODO
            for test_i, test_attrs in enumerate(value for key, value in test.items() if 'case' in key):
                # if test_i != 1: continue
                export_stride = test_attrs["export_stride"] if test_attrs.get("export_stride") else inp.export_stride
                controller.reset()
                smoke_attrs = test_attrs['smoke']
                sim.setup_world(
                    inp.simulation["re"],
                    inp.simulation['domain_size'],
                    inp.simulation['dt'],
                    inp.simulation['obs_mass'],
                    inp.simulation['obs_inertia'],
                    inp.simulation['reference_velocity'],
                    inp.simulation['sponge_intensity'],
                    inp.simulation['sponge_size'],
                    inp.simulation['inflow_on'],
                    smoke_attrs['buoyancy'] if smoke_attrs['on'] else (0, 0))
                if smoke_attrs['on']:
                    for xy in smoke_attrs['xy']:
                        sim.add_smoke(xy, smoke_attrs['width'], smoke_attrs['height'], smoke_attrs['inflow'])
                    if "smoke" not in export_vars: export_vars.append("smoke")
                control_force = torch.zeros(2).to(device)
                control_force_global = np.zeros(2)
                control_torque = torch.zeros(1).to(device)
                last_objective = test_attrs['positions'][0]
                for i in range(test_attrs['n_steps']):
                    # Get objective
                    x_objective_ = [objective for objective, i_objective in zip(test_attrs['positions'], test_attrs['i']) if i > i_objective][-1]
                    ang_objective_ = [objective for objective, i_objective in zip(test_attrs['angles'], test_attrs['i']) if i > i_objective][-1]
                    x_objective = torch.tensor(x_objective_).to(device)
                    ang_objective = torch.tensor(ang_objective_).to(device)
                    # Reset controller if objective changes
                    if last_objective != x_objective_:
                        last_objective = x_objective_
                        controller.reset()
                    add_forces = calculate_additional_forces(test_attrs.get('add_forces', {}), i).cpu().numpy()
                    sim.apply_forces(control_force_global + add_forces, control_torque)
                    sim.advect()
                    sim.make_incompressible()
                    sim.calculate_fluid_forces()
                    # Control
                    if i % sampling_stride == 0:
                        error_xy = (x_objective - sim.obstacle.geometry.center)
                        if not inp.translation_only:
                            angle_tensor = -(sim.obstacle.geometry.angle - math.PI / 2.0)
                            error_ang = -ang_objective - angle_tensor
                            control_effort = controller([np.array(error_xy.numpy()), np.array((error_ang.numpy(),))])
                            control_torque = control_effort[1]
                        else:
                            control_effort = controller([np.array(error_xy.numpy())])
                        # print(control_effort)
                        control_force = control_effort[0]
                        control_force_global = control_force
                    if math.any(sim.obstacle.geometry.center > inp.simulation['domain_size']) or math.any(sim.obstacle.geometry.center < (0, 0)):
                        break
                    # Time estimate
                    now = time.time()
                    delta = now - last
                    last = now
                    # Export
                    sim.reference_x = x_objective[0]
                    sim.reference_y = x_objective[1]
                    sim.error_x, sim.error_y = error_xy.native()
                    sim.add_forces_x, sim.add_forces_y = add_forces
                    sim.control_force_x, sim.control_force_y = control_force_global
                    if not inp.translation_only:
                        sim.reference_ang = ang_objective
                        sim.error_ang = -error_ang.native()
                        sim.control_torque = control_torque
                    # If not on stride just export scalar values
                    if (i % export_stride != 0):  # or (i < controller.past_window + 1):
                        export_vars_ = export_vars_scalar
                    else:
                        i_remaining = (len([key for key in test.keys() if 'case' in key]) - test_i - 1) * test_attrs['n_steps'] + (test_attrs['n_steps'] - i - 1)
                        remaining = i_remaining * delta
                        remaining_h = np.floor(remaining / 60. / 60.)
                        remaining_m = np.floor(remaining / 60. - remaining_h * 60.)
                        print(f"Time left: {remaining_h:.0f}h and {remaining_m:.0f} min - i: {i}")
                        export_vars_ = export_vars
                    sim.export_data(export_path, test_i, i, export_vars_ + ['add_forces_x', 'add_forces_y'], is_first_export)
                    is_first_export = False

    print("Done")

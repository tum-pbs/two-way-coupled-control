import time
import argparse
from Dataset import Dataset
from InputsManager import InputsManager
from misc_funcs import *
from InputsManager import RLInputsManager
from sac_actor import load_sac_torch_module
from sac_actor import SACActorModule
CUDA_LAUNCH_BLOCKING = 1
torch.set_printoptions(sci_mode=True)

if __name__ == "__main__":
    # ----------------------------------------------
    # -------------------- Inputs ------------------
    # ----------------------------------------------
    parser = argparse.ArgumentParser(description='Run test simulations with obstacle controlled by neural network model')
    parser.add_argument("runpaths", help="Path to folder containing model data")
    parser.add_argument("model_index", help="Index of model that will be used for tests")
    parser.add_argument("tests_id", nargs="+", help="ID of tests to be performed")
    args = parser.parse_args()
    model_id = int(args.model_index)
    run_path = args.runpaths
    tests_id = [f"test{i}" for i in args.tests_id]
    # Load inputs
    inp = InputsManager(os.path.abspath(run_path + "/inputs.json"))
    # Set model type
    if "online" in inp.__dict__.keys(): model_type = "online"
    elif "supervised" in inp.__dict__.keys(): model_type = "supervised"
    elif "rl" in inp.__dict__.keys(): model_type = "rl"
    else: raise ValueError("Unknown model type")
    # Set device
    if inp.device == "GPU":
        TORCH_BACKEND.set_default_device("GPU")
        device = torch.device("cuda:0")
    else:
        TORCH_BACKEND.set_default_device("CPU")
        device = torch.device("cpu")
    # Load tests json
    model_path = f"{run_path}/trained_model_{model_id:04d}.pth"
    tests = InputsManager(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json"), tests_id)  # TODO currently running only test1
    print(f"\n\n Running tests on model {model_path}")
    # max_error_xy = torch.as_tensor(200)
    for test_label, test in tests.__dict__.items():  # Loop through tests
        # ----------------------------------------------
        # ---------------- Setup simulation ------------
        # ----------------------------------------------
        export_path = f"{run_path}/tests/{test_label}_{model_id}/"
        print(f"\n Data will be saved on {export_path} \n")
        inp.delete_attributes(['export_stride', 'simulation'])
        inp.add_values(test["initial_conditions_path"] + "/inputs.json", ["simulation", "export_stride"])  # Load parameters of initial conditions
        # Create list of scalar variables that will be exported every step
        export_vars_scalar = list(inp.export_vars)
        for entry in list(export_vars_scalar):
            if entry in ["pressure", "vx", "vy", "obs_mask", "vorticity"]:
                export_vars_scalar.remove(entry)
            if "loss" in entry:
                export_vars_scalar.remove(entry)
                inp.export_vars.remove(entry)
        # Save tests used in this script
        tests.export(export_path + "tests.json", only=[test_label, "dataset_path", "tvt_ratio"])
        # Set simulation and probes parameters
        sim = TwoWayCouplingSimulation(inp.device, inp.translation_only)
        sim.set_initial_conditions(
            inp.simulation["obs_type"],
            inp.simulation['obs_width'],
            inp.simulation['obs_height'],
            test["initial_conditions_path"],
            obs_xy=inp.simulation['obs_xy'])
        probes = Probes(
            inp.simulation["obs_width"] / 2 + inp.probes_offset,
            inp.simulation['obs_height'] / 2 + inp.probes_offset,
            inp.probes_size,
            inp.probes_n_rows,
            inp.probes_n_columns,
            inp.simulation['obs_xy'])
        # ----------------------------------------------
        # ------------------- Setup NN -----------------
        # ----------------------------------------------
        sampling_stride = inp.training_dt / inp.simulation["dt"]  # In case model has a different sampling than simulation step
        assert sampling_stride.is_integer()
        # Load model
        if model_type == "rl":
            model = load_sac_torch_module(model_path).to(device)
        else:
            model = torch.load(os.path.abspath(model_path)).to(device)
        print("Model's state_dict:")  # Print model's attributes
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n Total amount of trainable parameters: {total_params}")
        # Dataset is used to get ground truth values if asked (useful for unstable models)
        # Make sure that the tests in tests.json came from this dataset
        ref_vars = inp.ref_vars
        # dataset = Dataset(tests.dataset_path, tests.tvt_ratio, ref_vars)
        # dataset.set_past_window_size(inp.past_window)
        # dataset.set_mode("validation")
        # Initialize inputs manager for reinforcement learning model
        if model_type == "rl":
            rl_inp = RLInputsManager(inp.past_window, inp.n_past_features, inp.rl['n_snapshots_per_window'], device)
        # Save a copy of the model that will be used for tests
        torch.save(model, os.path.abspath(f"{export_path}{model_path.split('/')[-1]}"))
        # ----------------------------------------------
        # ------------------ Simulate ------------------
        # ----------------------------------------------
        last_time = 0
        with torch.no_grad():
            is_first_export = True  # TODO Used for deleting previous files on folder
            for test_i, test_attrs in enumerate(value for key, value in test.items() if 'case' in key):
                export_stride = test_attrs["export_stride"] if test_attrs.get("export_stride") else inp.export_stride
                # Initialize tensors
                nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).to(device)
                control_force = torch.zeros(2).to(device)
                # control_force2 = torch.zeros(2).to(device)  # TODO
                control_force_global = torch.zeros(2).to(device)
                # control_force_global2 = torch.zeros(2).to(device)  # TODO
                control_torque = torch.zeros(1).to(device)
                # Initialize simulation
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
                    if "smoke" not in inp.export_vars: inp.export_vars.append("smoke")
                for i in range(test_attrs['n_steps']):
                    for x_objective_, ang_objective_, objective_i in zip(test_attrs['positions'], test_attrs['angles'], test_attrs['i']):
                        # Check if objective changed
                        if i > objective_i:
                            x_objective = torch.tensor(x_objective_).to(device)
                            ang_objective = torch.tensor(ang_objective_).to(device)
                    sim.apply_forces(control_force_global * ref_vars['force'], control_torque * ref_vars['torque'])
                    # sim.apply_forces((control_force_global + control_force_global2) * ref_vars['force'], (control_force[1] - control_force2[1]) * inp.simulation["obs_width"] / 2 * ref_vars['force'])
                    sim.advect()
                    sim.make_incompressible()
                    probes.update_transform(sim.obstacle.geometry.center.numpy(), -(sim.obstacle.geometry.angle.numpy() - math.PI / 2.0))
                    sim.calculate_fluid_forces()
                    # Control
                    if i % sampling_stride == 0:
                        nn_inputs_present, loss_inputs = extract_inputs(
                            inp.nn_vars,
                            sim,
                            probes,
                            x_objective,
                            ang_objective,
                            ref_vars,
                            inp.translation_only,
                            # clamp={'error_x': max_error_xy, 'error_y': max_error_xy}
                        )
                        if model_type == "rl":
                            rl_inp.add_snapshot(nn_inputs_present.view(1, -1))
                            # if i % inp.rl['n_snapshots_per_window'] == 0:
                            control_effort = model(rl_inp.values.view(1, -1))
                        else:
                            control_effort = model(nn_inputs_present.view(1, -1), nn_inputs_past.view(1, -1) if inp.past_window else None, inp.bypass_tanh)
                        # Warmup
                        if i < inp.past_window * sampling_stride: control_effort = torch.zeros_like(control_effort)
                        control_force = control_effort[0, :2]
                        control_force_global = control_force
                        if not inp.translation_only:
                            angle_tensor = -(sim.obstacle.geometry.angle - math.PI / 2.0).native()
                            control_force_global = rotate(control_force_global, angle_tensor)
                            control_torque = control_effort[0, -1:]
                        #     control_force2 = control_effort[0, 2:] * torch.as_tensor((0, 1)).cuda()  # TODO
                        #     control_force_global2 = rotate(control_force2, angle_tensor)  # Control force at global reference of frame (used for visualization only)

                        # Use ground truth values from dataset if needed
                        # if i < test_attrs['help_i'] and inp.translation_only:
                        #     if i < inp.past_window:
                        #         _, gt_inputs_past, _, indexes = dataset.get_values_by_case_snapshot(test_i, [inp.past_window])
                        #         control_force_global = gt_inputs_past[0][[[indexes["control_force_x"][i], indexes["control_force_y"][i]]]]
                        #     else:
                        #         gt_inputs_present, gt_inputs_past, gt_force, indexes = dataset.get_values_by_case_snapshot(test_i, [i])
                        #         control_force_global = gt_force.view(-1)
                        if model_type in ["supervised", "online"]:
                            nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_effort)
                    # Stop simulation if obstacle escapes domain
                    if math.any(sim.obstacle.geometry.center > inp.simulation['domain_size']) or math.any(sim.obstacle.geometry.center < (0, 0)):
                        break
                    # Time estimate variables
                    now = time.time()
                    delta = now - last_time
                    last_time = now
                    # ----------------------------------------------
                    # ------------------- Export -------------------
                    # ----------------------------------------------
                    probes_points = probes.get_points_as_tensor()
                    sim.probes_vx = sim.velocity.x.sample_at(probes_points).native().detach()
                    sim.probes_vy = sim.velocity.y.sample_at(probes_points).native().detach()
                    sim.probes_points = probes_points.native().detach()
                    sim.reference_x = x_objective[0].detach()
                    sim.reference_y = x_objective[1].detach()
                    sim.control_force_x, sim.control_force_y = control_force_global.detach() * ref_vars['force']
                    sim.error_x = loss_inputs['error_x'].detach() * ref_vars['length']
                    sim.error_y = loss_inputs['error_y'].detach() * ref_vars['length']
                    if not inp.translation_only:
                        sim.error_x, sim.error_y = rotate(torch.tensor([sim.error_x, sim.error_y]), angle_tensor)
                        sim.reference_ang = ang_objective.detach()
                        sim.error_ang = loss_inputs['error_ang'].detach() * ref_vars['angle']
                        # sim.control_force_x2, sim.control_force_y2 = control_force_global2.detach() * ref_vars['force']  # TODO
                        sim.control_torque = control_torque.detach() * ref_vars['torque']
                    # If not on stride export just scalar values
                    if (i % export_stride != 0):  # or (i < inp.past_window + 1):
                        export_vars = export_vars_scalar
                    else:
                        export_vars = inp.export_vars
                        # Print remaining time
                        i_remaining = (len([key for key in test.keys() if 'case' in key]) - test_i - 1) * test_attrs['n_steps'] + (test_attrs['n_steps'] - i - 1)
                        remaining = i_remaining * delta
                        remaining_h = np.floor(remaining / 60. / 60.)
                        remaining_m = np.floor(remaining / 60. - remaining_h * 60.)
                        print(f"Time left: {remaining_h:.0f}h and {remaining_m:.0f} min")
                    # export_vars = ['vorticity', 'obs_mask', 'obs_xy', 'control_force_x', 'control_force_y', 'fluid_force_x', 'fluid_force_y', 'reference_x', 'reference_y', 'error_x', 'error_y', ]
                    sim.export_data(export_path, test_i, i, export_vars, is_first_export)
                    is_first_export = False
    print("Done")

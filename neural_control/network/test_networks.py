import time
import argparse
from Dataset import Dataset
from InputsManager import InputsManager
from misc_funcs import *
CUDA_LAUNCH_BLOCKING = 1
torch.set_printoptions(sci_mode=True)
if __name__ == "__main__":
    # inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json")
    # inp.calculate_properties()

    # ----------------------------------------------
    # -------------------- Inputs ------------------
    # ----------------------------------------------
    parser = argparse.ArgumentParser(description='Run test simulations with obstacle controlled by model')
    parser.add_argument("runpaths", nargs="+", help="Path to folders containing model data")
    parser.add_argument("model_index", help="Index of model that will be used for tests")
    args = parser.parse_args()
    model_id = int(args.model_index)
    run_paths = args.runpaths
    for run_path in run_paths:
        inp = InputsManager(os.path.abspath(run_path + "/inputs.json"), exclude=["simulation"])
        if inp.device == "GPU":
            TORCH_BACKEND.set_default_device("GPU")
            device = torch.device("cuda:0")
        else:
            TORCH_BACKEND.set_default_device("CPU")
            device = torch.device("cpu")
        model_path = f"{run_path}/trained_model_{model_id:04d}.pth"
        print(f"\n\n Running tests on model {model_path}")
        # Load tests json
        tests = InputsManager(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json"), ["test1"])  # TODO should run all tests
        tests.model_path = model_path
        for test_label, test in [(key, value) for key, value in tests.__dict__.items() if isinstance(value, dict)]:  # Loop through tests
            # ----------------------------------------------
            # ---------------- Setup simulation ------------
            # ----------------------------------------------
            inp.add_values(test["initial_conditions_path"] + "/inputs.json", ["simulation"])  # Load parameters of initial conditions
            # Create list of scalar variables that will be exported every step
            export_vars_scalar = list(inp.export_vars)
            for entry in list(export_vars_scalar):
                if entry in ["pressure", "vx", "vy", "obs_mask", "vorticity"]:
                    export_vars_scalar.remove(entry)
                if "loss" in entry:
                    export_vars_scalar.remove(entry)
                    inp.export_vars.remove(entry)
            # Save tests used in this script
            export_path = f"{run_path}/tests/{test_label}_{model_id}/"
            print(f"\n Data will be saved on {export_path} \n")
            prepare_export_folder(export_path, 0)
            tests.export(export_path + "tests.json", only=[test_label, "dataset_path", "tvt_ratio"])
            sim = TwoWayCouplingSimulation(inp.device, inp.translation_only)
            sim.set_initial_conditions(
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
            ref_vars = dict(
                velocity=inp.simulation['inflow_velocity'],
                length=inp.simulation['obs_width'],
                force=inp.simulation['obs_mass'] * inp.max_acc,
                angle=PI,
                torque=inp.simulation['obs_inertia'] * inp.max_ang_acc,
                time=inp.simulation['obs_width'] / inp.simulation['inflow_velocity'],
                ang_velocity=inp.simulation['inflow_velocity'] / inp.simulation['obs_width']
            )
            last = 0
            model = torch.load(os.path.abspath(model_path)).to(device)
            print("Model's state_dict:")
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n Total amount of trainable parameters: {total_params}")
            # Dataset used is used to get objectives and ground truth values if demanded
            # dataset = Dataset(tests.dataset_path, tests.tvt_ratio, ref_vars)
            # dataset.set_past_window_size(inp.past_window)
            # dataset.set_mode("validation")
            torch.save(model, os.path.abspath(f"{export_path}{model_path.split('/')[-1]}"))
            # ----------------------------------------------
            # ------------------ Simulate ------------------
            # ----------------------------------------------
            with torch.no_grad():
                is_first_export = True  # Used for deleting previous files on folder
                for test_i, test_attrs in enumerate(value for key, value in test.items() if 'test' in key):
                    sim.setup_world(
                        inp.simulation["re"],
                        inp.simulation['domain_size'],
                        inp.simulation['dt'],
                        inp.simulation['obs_mass'],
                        inp.simulation['obs_inertia'],
                        inp.simulation['inflow_velocity'],
                        inp.simulation['sponge_intensity'],
                        inp.simulation['sponge_size'],)
                    nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).to(device)
                    control_force = torch.zeros(2).to(device)
                    control_force_global = torch.zeros(2).to(device)
                    control_torque = torch.zeros(1).to(device)
                    integrator_force = 0
                    integrator_torque = 0
                    for i in range(1, test_attrs['n_steps'] + 1):
                        for x_objective_, ang_objective_, objective_i in zip(test_attrs['positions'], test_attrs['angles'], test_attrs['i']):
                            # Check if objective changed
                            if i > objective_i:
                                x_objective = torch.tensor(x_objective_).to(device)
                                ang_objective = torch.tensor(ang_objective_).to(device)
                        sim.apply_forces(control_force_global * ref_vars['force'], control_torque * ref_vars['torque'])
                        sim.advect()
                        sim.make_incompressible()
                        probes.update_transform(sim.obstacle.geometry.center.numpy(), -(sim.obstacle.geometry.angle.numpy() - math.PI / 2.0))
                        sim.calculate_fluid_forces()
                        # Control
                        nn_inputs_present, loss_inputs_present = extract_inputs(sim, probes, x_objective, ang_objective, ref_vars, inp.translation_only)
                        control_effort = model(nn_inputs_present, nn_inputs_past if inp.past_window else None)
                        control_effort = torch.clamp(control_effort, -1., 1.)
                        # integrator_force = integrator_force + tests.integrator_force_theta * loss_inputs_present[0, :2]
                        control_force = control_effort[0, :2] + integrator_force
                        angle_tensor = -(sim.obstacle.geometry.angle - math.PI / 2.0).native()
                        control_force_global = rotate(control_force, angle_tensor)
                        # Help stage
                        if i < test_attrs['help_i'] and inp.translation_only:
                            if i <= inp.past_window:
                                dataset_i = i - (1)
                                # _, gt_inputs_past, _, indexes = dataset.get_values_by_case_snapshot(test_i, [dataset_i])
                                # control_force_global = gt_inputs_past[0][[[indexes["control_force_x"][0], indexes["control_force_y"][0]]]]
                            else:
                                dataset_i = i - (1) - inp.past_window
                                gt_inputs_present, gt_inputs_past, gt_force, indexes = dataset.get_values_by_case_snapshot(test_i, [dataset_i])
                                control_force_global = gt_force.view(-1)
                            control_effort = control_force_global.view(-1)
                            print("Using GT values")
                        if not inp.translation_only:
                            control_torque = control_effort[0, -1:]  # + integrator_torque
                        nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_effort)
                        if math.any(sim.obstacle.geometry.center > inp.simulation['domain_size']) or math.any(sim.obstacle.geometry.center < (0, 0)):
                            break
                        # Time estimate
                        now = time.time()
                        delta = now - last
                        last = now
                        # Export
                        probes_points = probes.get_points_as_tensor()
                        sim.probes_vx = sim.velocity.x.sample_at(probes_points).native().detach()
                        sim.probes_vy = sim.velocity.y.sample_at(probes_points).native().detach()
                        sim.probes_points = probes_points.native().detach()
                        sim.reference_x = x_objective[0].detach().clone()
                        sim.reference_y = x_objective[1].detach().clone()
                        sim.error_x, sim.error_y = rotate(loss_inputs_present[0, :2].detach().clone() * ref_vars['length'], angle_tensor)
                        sim.control_force_x, sim.control_force_y = control_force_global.detach().clone() * ref_vars['force']  # XXX
                        if not inp.translation_only:
                            sim.reference_angle = ang_objective.detach().clone()
                            sim.error_ang = loss_inputs_present[0, 4].detach().clone() * ref_vars['angle']
                            sim.control_torque = control_torque.detach().clone() * ref_vars['torque']
                        # If not on stride just export scalar values
                        if (i % inp.export_stride != 0) or (i < inp.past_window + 1):
                            export_vars = export_vars_scalar
                        else:
                            i_remaining = (len([key for key in test.keys() if 'test' in key]) - test_i - 1) * test_attrs['n_steps'] + (test_attrs['n_steps'] - i - 1)
                            remaining = i_remaining * delta
                            remaining_h = np.floor(remaining / 60. / 60.)
                            remaining_m = np.floor(remaining / 60. - remaining_h * 60.)
                            print(f"Time left: {remaining_h:.0f}h and {remaining_m:.0f} min")
                            export_vars = inp.export_vars
                        sim.export_data(export_path, test_i, i, export_vars, is_first_export)
                        is_first_export = False
                        if i % inp.export_stride != 0:
                            continue

import time
import argparse
from neural_control.InputsManager import InputsManager, RLInputsManager
from neural_control.misc.misc_funcs import *
from neural_control.neural_networks.NeuralController import NeuralController
from neural_control.neural_networks.rl.extract_model import load_sac_torch_module, SACActorModule
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
    parser.add_argument("-onesimulation", action="store_true", help="Use this flag to run only one test simulation")
    args = parser.parse_args()
    model_id = int(args.model_index)
    run_path = args.runpaths
    flag_onesim = args.onesimulation
    if flag_onesim: print("\n Running only one simulation \n")
    tests_id = [f"test{i}" for i in args.tests_id]
    # Load inputs
    inp = InputsManager(os.path.abspath(run_path + "/inputs.json"))
    # Set model type
    if "unsupervised" in inp.__dict__.keys(): model_type = "unsupervised"
    elif "supervised" in inp.__dict__.keys(): model_type = "supervised"
    elif "rl" in inp.__dict__.keys(): model_type = "rl"
    else: raise ValueError("Unknown model type")
    # Set device
    if inp.device == "GPU" and torch.cuda.is_available():
        TORCH_BACKEND.set_default_device("GPU")
        device = torch.device("cuda:0")
    else:
        TORCH_BACKEND.set_default_device("CPU")
        device = torch.device("cpu")
    # Load tests json
    model_path = f"{run_path}/trained_model_{model_id:04d}.pt"
    tests = InputsManager(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json"), tests_id)
    print(f"\n\n Running tests on model {model_path}")
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
        # Set simulation parameters
        sim = TwoWayCouplingSimulation(inp.device, inp.translation_only)
        sim.set_initial_conditions(
            inp.simulation["obs_type"],
            inp.simulation['obs_width'],
            inp.simulation['obs_height'],
            test["initial_conditions_path"],
            obs_xy=inp.simulation['obs_xy'])
        # ----------------------------------------------
        # ------------------- Setup NN -----------------
        # ----------------------------------------------
        sampling_stride = inp.training_dt / inp.simulation["dt"]  # In case model has a different sampling than simulation step
        assert sampling_stride.is_integer()
        # Load model
        if model_type == "rl":
            latent_pi = torch.nn.Sequential(
                torch.nn.Linear(16, 38),
                torch.nn.ReLU(),
                torch.nn.Linear(38, 38),
                torch.nn.ReLU())
            mu = torch.nn.Linear(38, 2)
            model = SACActorModule(latent_pi, mu, 0, 0, (16,)).to(device)
        else:
            model = NeuralController(
                f"{inp.architecture}{inp.past_window}",
                2 if inp.translation_only else 3,
                inp.n_present_features,
                inp.n_past_features,
                inp.past_window).to(device)
        model.load_state_dict(torch.load(os.path.abspath(model_path), map_location=device))
        print("Model's state_dict:")  # Print model's attributes
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n Total amount of trainable parameters: {total_params}")
        # Make sure that the tests in tests.json came from this dataset
        ref_vars = inp.ref_vars
        # Initialize inputs manager for reinforcement learning model
        if model_type == "rl":
            rl_inp = RLInputsManager(inp.past_window, inp.n_past_features, inp.rl['n_snapshots_per_window'], device)
        # Save a copy of the model that will be used for tests
        torch.save(model.state_dict(), os.path.abspath(f"{export_path}{model_path.split('/')[-1]}"))
        # ----------------------------------------------
        # ------------------ Simulate ------------------
        # ----------------------------------------------
        last_time = 0
        with torch.no_grad():
            is_first_export = True  # Used for deleting previous files on folder
            for test_i, test_attrs in enumerate(value for key, value in test.items() if 'case' in key):
                export_stride = test_attrs["export_stride"] if test_attrs.get("export_stride") else inp.export_stride
                # Initialize tensors
                nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).to(device)
                control_force = torch.zeros(2).to(device)
                control_force_global = torch.zeros(2).to(device)
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
                    add_forces = calculate_additional_forces(test_attrs.get('add_forces', {}), i).to(device)
                    sim.apply_forces(control_force_global * ref_vars['force'] + add_forces, control_torque * ref_vars['torque'])
                    sim.advect()
                    sim.make_incompressible()
                    sim.calculate_fluid_forces()
                    # Control
                    if i % sampling_stride == 0:
                        nn_inputs_present, loss_inputs = extract_inputs(
                            inp.nn_vars,
                            sim,
                            x_objective,
                            ang_objective,
                            ref_vars,
                            inp.translation_only,
                        )
                        if model_type == "rl":
                            inputs = torch.cat([nn_inputs_past.view(1, -1), nn_inputs_present.view(1, -1)], dim=1)
                            control_effort = model(inputs)
                        else:
                            control_effort = model(
                                nn_inputs_present.view(1, -1),
                                nn_inputs_past.view(1, -1) if inp.past_window else None,
                                inp.bypass_tanh)
                        # Warmup
                        if i < inp.past_window * sampling_stride: control_effort = torch.zeros_like(control_effort)
                        control_force = control_effort[0, :2]
                        control_force_global = control_force
                        if not inp.translation_only:
                            angle_tensor = -(sim.obstacle.geometry.angle - math.PI / 2.0).native()
                            control_force_global = rotate(control_force_global, angle_tensor)
                            control_torque = control_effort[0, -1:]
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
                    sim.reference_x = x_objective[0].detach()
                    sim.reference_y = x_objective[1].detach()
                    sim.control_force_x, sim.control_force_y = control_force_global.detach() * ref_vars['force']
                    sim.add_forces_x, sim.add_forces_y = add_forces.detach()
                    sim.error_x = loss_inputs['error_x'].detach() * ref_vars['length']
                    sim.error_y = loss_inputs['error_y'].detach() * ref_vars['length']
                    if not inp.translation_only:
                        sim.error_x, sim.error_y = rotate(torch.tensor([sim.error_x, sim.error_y]), angle_tensor)
                        sim.reference_ang = ang_objective.detach()
                        sim.error_ang = loss_inputs['error_ang'].detach() * ref_vars['angle']
                        sim.control_torque = control_torque.detach() * ref_vars['torque']
                    # If not on stride export just scalar values
                    if (i % export_stride != 0):  # or (i < inp.past_window + 1):
                        export_vars = export_vars_scalar
                    else:
                        export_vars = inp.export_vars
                        # Print remaining time
                        tests_remaining = len([key for key in test.keys() if 'case' in key]) - test_i - 1
                        i_remaining = tests_remaining * test_attrs['n_steps'] * (1 - flag_onesim) + (test_attrs['n_steps'] - i - 1)
                        remaining = i_remaining * delta
                        remaining_h = np.floor(remaining / 60. / 60.)
                        remaining_m = np.floor(remaining / 60. - remaining_h * 60.)
                        print(f"Time left: {remaining_h:.0f}h and {remaining_m:.0f} min - i: {i}")
                    sim.export_data(export_path, test_i, i, export_vars + ['add_forces_x', 'add_forces_y'], is_first_export)
                    is_first_export = False
                if flag_onesim: break
    print("Done")

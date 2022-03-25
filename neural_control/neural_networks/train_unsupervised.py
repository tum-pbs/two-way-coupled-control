from math import ceil
from neural_control.InputsManager import InputsManager
from neural_control.neural_networks.NeuralController import NeuralController
from neural_control.misc.misc_funcs import *
import argparse
import torch.utils.tensorboard as tb
from time import time


if __name__ == "__main__":
    # ----------------------------------------------
    # -------------------- Inputs ------------------
    # ----------------------------------------------
    seed = 100
    inp = InputsManager(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json"), ['unsupervised'])
    inp.calculate_properties()
    inp.add_values(os.path.abspath(inp.unsupervised['simulation_path'] + 'inputs.json'), ["simulation"])  # Load simulation inputs
    if inp.device == "GPU":
        TORCH_BACKEND.set_default_device("GPU")
        device = torch.device("cuda:0")
    else:
        TORCH_BACKEND.set_default_device("CPU")
        device = torch.device("cpu")
    parser = argparse.ArgumentParser(description='Train nn in an unsupervised setting')
    parser.add_argument("export_path", help="data will be saved in this path")
    args = parser.parse_args()
    inp.export_path = args.export_path + '/'
    # ----------------------------------------------
    # ---------------- Setup simulation ------------
    # ----------------------------------------------
    sim = TwoWayCouplingSimulation(inp.device, inp.translation_only)
    sim.set_initial_conditions(inp.simulation['obs_type'], inp.simulation['obs_width'], inp.simulation['obs_height'], path=os.path.abspath(inp.unsupervised['simulation_path']))
    ref_vars = dict(
        velocity=inp.simulation['reference_velocity'],
        length=inp.simulation['reference_length'],
        force=inp.simulation['obs_mass'] * inp.max_acc,
        angle=PI,
        torque=inp.simulation['obs_inertia'] * inp.max_ang_acc,
        time=inp.simulation['obs_width'] / inp.simulation['reference_velocity'],
        ang_velocity=inp.simulation['reference_velocity'] / inp.simulation['obs_width']
    )
    inp.ref_vars = ref_vars
    destinations_zone_size = inp.simulation['domain_size'] - inp.unsupervised['destinations_margins'] * 2
    # ----------------------------------------------
    # ------------------- Setup NN -----------------
    # ----------------------------------------------
    inp.training_dt = inp.simulation['dt']
    if inp.resume_training:  # TODO
        # raise NotImplementedError
        # Load most trained model
        model_file = natsorted(name for name in os.listdir(os.path.abspath(inp.export_path)) if ".pth" in name)
        print(f"Loading model {model_file[-1]}")
        first_case = int(model_file[-1].split("trained_model_")[-1][:4]) + 1
        model = torch.load(inp.export_path + model_file[-1])
    else:
        first_case = 0
        torch.manual_seed(seed)
        model = NeuralController(
            f"{inp.architecture}{inp.past_window}",
            2 if inp.translation_only else 3,
            inp.n_present_features,
            inp.n_past_features,
            inp.past_window)
    torch.manual_seed(seed + 1)
    model.to(device)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total amount of trainable parameters: {total_params}")
    inp.unsupervised["n_params"] = total_params
    optimizer_func = getattr(torch.optim, inp.unsupervised['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=inp.unsupervised['learning_rate'])
    decay = np.exp(np.log(0.5) / inp.unsupervised['lr_half_life'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay)
    # ----------------------------------------------
    # ---------- Pre training processing -----------
    # ----------------------------------------------
    # Prepare folder for saving data
    shutil.rmtree(inp.export_path + 'tensorboard', ignore_errors=True)
    writer = tb.SummaryWriter(inp.export_path + '/tensorboard/')
    torch.save(model.state_dict(), os.path.abspath(f"{inp.export_path}/initial_model_{first_case}.pt"))
    # Number of simulations necessary to achieve desired number of iterations
    n_simulations = inp.unsupervised["n_iterations"] / (inp.unsupervised["n_timesteps"] / inp.unsupervised["n_before_backprop"])
    n_simulations = ceil(n_simulations / inp.unsupervised["simulation_dropout"]) * 2  # Add double the amount necessary just in case some dont converge
    # Create objectives
    torch.manual_seed(seed + 2)
    xy = torch.rand(2, n_simulations)
    ang = torch.rand(n_simulations)
    for case in range(n_simulations):
        growrate = np.min((inp.unsupervised['destinations_zone_growrate'] * (case + 1), 1.))
        margins = (inp.simulation['domain_size'] - destinations_zone_size * growrate) / 2
        xy[:, case] = xy[:, case] * destinations_zone_size * growrate + margins
        ang[case] = ((ang[case] * 2 * PI - PI) * growrate)
    # Save objectives
    inp.unsupervised['objective_xy'] = xy.numpy().T.tolist()
    inp.unsupervised['objective_ang'] = ang.numpy().tolist()
    objective_xy = xy.to(device)
    objective_ang = ang.to(device)
    inp.export(inp.export_path + "inputs.json", exclude='supervised_training')
    # ----------------------------------------------
    # ----------------- Simulation -----------------
    # ----------------------------------------------
    last_time = time()
    i_bp = 0
    case = 0
    while i_bp < inp.unsupervised['n_iterations']:
        # Setup case with default initial values
        sim.setup_world(
            inp.simulation['re'],
            inp.simulation['domain_size'],
            inp.simulation['dt'],
            inp.simulation['obs_mass'],
            inp.simulation['obs_inertia'],
            inp.simulation['reference_velocity'],
            inp.simulation['sponge_intensity'],
            inp.simulation['sponge_size'],
            inp.simulation['inflow_on'])
        # Variables initialization
        nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).to(device)
        loss_inputs = defaultdict(lambda: torch.zeros((inp.unsupervised['n_before_backprop'], 1, 1)).to(device))
        if inp.translation_only:
            control_effort = torch.zeros(2).to(device)
        else:
            control_effort = torch.zeros(3).to(device)
        last_control_force = torch.zeros(2).to(device)
        control_force_global = torch.zeros(2).to(device)
        control_force = torch.zeros(2).to(device)
        control_force2 = torch.zeros(2).to(device)
        control_torque = torch.zeros(1).to(device)
        last_control_torque = torch.zeros(1).to(device)
        loss = torch.zeros(1).to(device)
        loss_terms = defaultdict(lambda: 0)
        # Run simulation
        for i in range(0, inp.unsupervised['n_timesteps'] + 1):
            # Check CFL in the first iterations due to possible numerical instabilities
            if case < 10:
                if math.max(math.abs(sim.velocity.values)) * sim.dt > 1.5:
                    print("CFL too big. Resetting simulation")
                    break
            sim.apply_forces(control_force_global * ref_vars['force'], control_torque * ref_vars['torque'])
            sim.advect()
            # In case obstacle escapes domain
            if math.any(sim.obstacle.geometry.center > inp.simulation['domain_size']) or math.any(sim.obstacle.geometry.center < (0, 0)):
                print("Obstacle is out of bounds. Resetting simulation")
                break
            sim.make_incompressible()
            sim.calculate_fluid_forces()
            # Control
            nn_inputs_present, loss_inputs_present = extract_inputs(inp.nn_vars, sim, objective_xy[:, case], objective_ang[case], ref_vars, inp.translation_only)
            if i < inp.past_window: last_backprop = i  # Wait to backprop until past inputs are cached
            else:
                control_effort = model(nn_inputs_present.view(1, -1), nn_inputs_past.view(1, -1))
                control_force = control_effort[0, :2]
                loss_inputs_present['d_control_force_x'], loss_inputs_present['d_control_force_y'] = (control_force - last_control_force)
                loss_inputs_present['control_force_x'], loss_inputs_present['control_force_y'] = control_force
                last_control_force = control_force
                if inp.translation_only:
                    control_force_global = control_force
                else:
                    angle_tensor = -(sim.obstacle.geometry.angle - math.PI / 2.0).native()
                    control_force_global = rotate(control_force, angle_tensor)  # Control force at global reference of frame (used for visualization only)
                    # Additional inputs for loss
                    control_torque = control_effort[0, -1:]
                    d_control_torque = control_torque - last_control_torque
                    last_control_torque = control_torque
                    loss_inputs['d_control_torque'] = d_control_torque
                    loss_inputs['control_torque'] = control_torque
                # Save quantities necessary for loss
                for key in loss_inputs_present:
                    loss_inputs[key] = torch.cat((loss_inputs[key][1:, ...], loss_inputs_present[key].view(1, 1, 1)))
                if (i - last_backprop == inp.unsupervised['n_before_backprop']):
                    loss, loss_terms = calculate_loss(loss_inputs, inp.unsupervised['hyperparams'], inp.translation_only)
                    loss.backward()
                    optimizer.step()
                    i_bp += 1
                    if i_bp % inp.unsupervised["model_export_stride"] == 0:
                        model_id = int(i_bp / inp.unsupervised["model_export_stride"])
                        print(f"Saving model {model_id}")
                        torch.save(model.state_dict(), os.path.abspath(f"{inp.export_path}/trained_model_{model_id:04d}.pt"))
                        weigths, biases = get_weights_and_biases(model)
                        for var in (weigths, biases):
                            for tag, value in var.items():
                                writer.add_histogram(tag, value, global_step=i_bp)
                        writer.flush()
                    if i_bp == inp.unsupervised['n_iterations']: break
                    lr_scheduler.step()  # Decay learning rate after backprop
                    optimizer.zero_grad()
                    # Reset graph
                    control_force = None
                    loss_inputs = {key: value.detach() for key, value in loss_inputs.items()}
                    control_effort = control_effort.detach()
                    control_torque = control_torque.detach()
                    last_control_force = last_control_force.detach()
                    last_control_torque = last_control_torque.detach()
                    nn_inputs_present = nn_inputs_present.detach()
                    nn_inputs_past = nn_inputs_past.detach()
                    control_force_global = control_force_global.detach()
                    sim.detach_variables()
                    last_backprop = i
            nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_effort)
            # Export values
            if (i % inp.export_stride != 0): continue
            sim.loss = loss.detach()
            for key, value in loss_terms.items():
                setattr(sim, f"loss_{key}", value.detach())
            sim.reference_x = objective_xy[0, case].detach()
            sim.reference_y = objective_xy[1, case].detach()
            sim.control_force_x, sim.control_force_y = control_force_global.detach() * ref_vars['force']
            sim.error_x = loss_inputs_present['error_x'].detach() * ref_vars['length']
            sim.error_y = loss_inputs_present['error_y'].detach() * ref_vars['length']
            if not inp.translation_only:
                angle = -(sim.obstacle.geometry.angle - math.PI / 2.0).native().detach()
                sim.error_x, sim.error_y = rotate(torch.cat((sim.error_x, sim.error_y)), angle)
                sim.reference_ang = objective_ang[case].detach()
                sim.error_ang = loss_inputs_present['error_ang'].detach() * ref_vars['angle']
                sim.control_torque = control_torque.detach() * ref_vars['torque']
            sim.export_data(inp.export_path, case, int(i / inp.export_stride), inp.export_vars, (case == 0 and i == 0))
            # Calculate how much time is left
            current_time = time()
            steps_left = (((n_simulations / 2 - 1) - (case + 1)) * inp.unsupervised['n_timesteps'] + inp.unsupervised['n_timesteps'] - (i + 1)) / inp.export_stride
            time_left = steps_left * (current_time - last_time) / 3600
            last_time = current_time
            time_left_hours = int(time_left)
            time_left_minutes = int((time_left - time_left_hours) * 60)
            print(f"Time left: {time_left_hours:d}h {time_left_minutes:d} min")
            i += 1
        case += 1

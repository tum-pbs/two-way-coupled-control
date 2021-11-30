from math import ceil
from InputsManager import InputsManager
# import torch.utils.tensorboard as tb
from NeuralController import NeuralController
from misc_funcs import *
import argparse
import torch.utils.tensorboard as tb
from time import time


if __name__ == "__main__":
    # ----------------------------------------------
    # -------------------- Inputs ------------------
    # ----------------------------------------------
    inp = InputsManager(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json"), ['online'])
    inp.calculate_properties()
    inp.add_values(os.path.abspath(inp.online['simulation_path'] + 'inputs.json'), ["probes", "simulation"])  # Load simulation inputs
    if inp.device == "GPU":
        TORCH_BACKEND.set_default_device("GPU")
        device = torch.device("cuda:0")
    else:
        TORCH_BACKEND.set_default_device("CPU")
        device = torch.device("cpu")
    parser = argparse.ArgumentParser(description='Train nn in an online setting')
    parser.add_argument("export_path", help="data will be saved in this path")
    args = parser.parse_args()
    inp.export_path = args.export_path + '/'
    # ----------------------------------------------
    # ---------------- Setup simulation ------------
    # ----------------------------------------------
    sim = TwoWayCouplingSimulation(inp.device, inp.translation_only)
    sim.set_initial_conditions(inp.simulation['obs_type'], inp.simulation['obs_width'], inp.simulation['obs_height'], path=os.path.abspath(inp.online['simulation_path']))
    probes = Probes(
        inp.simulation['obs_width'] / 2 + inp.probes_offset,
        inp.simulation['obs_height'] / 2 + inp.probes_offset,
        inp.probes_size,
        inp.probes_n_rows,
        inp.probes_n_columns,
        inp.simulation['obs_xy'])
    ref_vars = dict(
        velocity=inp.simulation['reference_velocity'],
        length=inp.simulation['obs_width'],
        force=inp.simulation['obs_mass'] * inp.max_acc,
        angle=PI,
        torque=inp.simulation['obs_inertia'] * inp.max_ang_acc,
        time=inp.simulation['obs_width'] / inp.simulation['reference_velocity'],
        ang_velocity=inp.simulation['reference_velocity'] / inp.simulation['obs_width']
    )
    inp.ref_vars = ref_vars
    destinations_zone_size = inp.simulation['domain_size'] - inp.online['destinations_margins'] * 2
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
        model = NeuralController(
            f"{inp.architecture}{inp.past_window}",
            2 if inp.translation_only else 3,  # TODO
            # 2 if inp.translation_only else 4,  # TODO
            inp.n_present_features,
            inp.n_past_features,
            inp.past_window)
    torch.manual_seed(999)
    model.to(device)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total amount of trainable parameters: {total_params}")
    inp.online["n_params"] = total_params
    optimizer_func = getattr(torch.optim, inp.online['optimizer'])
    optimizer = optimizer_func(model.parameters(), lr=inp.online['learning_rate'])
    decay = np.exp(np.log(0.5) / inp.learning_rate_decay_half_life)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay)
    # ----------------------------------------------
    # ---------- Pre training processing -----------
    # ----------------------------------------------
    # Prepare folder for saving data
    prepare_export_folder(inp.export_path, first_case)
    shutil.rmtree(inp.export_path + 'tensorboard', ignore_errors=True)
    writer = tb.SummaryWriter(inp.export_path + '/tensorboard/')
    torch.save(model, os.path.abspath(f"{inp.export_path}/initial_model_{first_case}.pth"))
    # Number of simulations necessary to achieve desired number of iterations
    n_simulations = inp.online["n_iterations"] / (inp.online["n_timesteps"] / inp.online["n_before_backprop"])
    n_simulations = ceil(n_simulations / inp.online["simulation_dropout"]) + 2  # Add two to make sure we have enough simulations
    # Gradually increase objectives distances
    # Create objectives
    torch.manual_seed(900)
    xy = torch.rand(2, n_simulations)
    ang = torch.rand(n_simulations)
    for case in range(n_simulations):
        growrate = np.min((inp.online['destinations_zone_growrate'] * (case + 1), 1.))
        margins = (inp.simulation['domain_size'] - destinations_zone_size * growrate) / 2
        xy[:, case] = xy[:, case] * destinations_zone_size * growrate + margins
        ang[case] = ((ang[case] * 2 * PI - PI) * growrate)
    # Save objectives
    inp.online['objective_xy'] = xy.numpy().T.tolist()
    inp.online['objective_ang'] = ang.numpy().tolist()
    objective_xy = xy.to(device)
    objective_ang = ang.to(device)
    inp.export(inp.export_path + "inputs.json", exclude='supervised_training')
    # ----------------------------------------------
    # ----------------- Simulation -----------------
    # ----------------------------------------------
    last_time = time()
    # for case in range(first_case, inp.online['n_cases']):
    i_bp = 0
    case = 0
    while i_bp < inp.online['n_iterations']:
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
        if inp.translation_only:
            control_effort = torch.zeros(2).to(device)
        else:
            # loss_inputs = torch.zeros((inp.online['n_before_backprop'], 1, 10)).to(device)  # TODO
            control_effort = torch.zeros(3).to(device)
            # control_effort = torch.zeros(4).to(device)  # TODO
        last_control_force = torch.zeros(2).to(device)
        # last_control_force2 = torch.zeros(2).to(device)  # TODO
        control_force_global = torch.zeros(2).to(device)
        # control_force_global2 = torch.zeros(2).to(device)  # TODO
        control_force = torch.zeros(2).to(device)
        control_force2 = torch.zeros(2).to(device)
        control_torque = torch.zeros(1).to(device)
        last_control_torque = torch.zeros(1).to(device)
        loss = torch.zeros(1).to(device)
        loss_terms = torch.zeros((5)).to(device)
        # Run simulation
        for i in range(1, inp.online['n_timesteps'] + 1):
            sim.apply_forces(control_force_global * ref_vars['force'], control_torque * ref_vars['torque'])
            # sim.apply_forces((control_force_global + control_force_global2) * ref_vars['force'], (control_force[1] - control_force2[1]) * inp.simulation["obs_width"] / 2 * ref_vars['force'])
            sim.advect()
            if math.any(sim.obstacle.geometry.center > inp.simulation['domain_size']) or math.any(sim.obstacle.geometry.center < (0, 0)): break  # In case obstacle escapes domain
            sim.make_incompressible()
            # probes.update_transform(sim.obstacle.geometry.center.numpy(), -(sim.obstacle.geometry.angle.numpy() - math.PI / 2.0)) # TODO
            sim.calculate_fluid_forces()
            # Control
            nn_inputs_present, loss_inputs = extract_inputs(inp.nn_vars, sim, probes, objective_xy[:, case], objective_ang[case], ref_vars, inp.translation_only)
            if i < inp.past_window + 1: last_backprop = i  # Wait to backprop until past inputs are cached
            else:
                control_effort = model(nn_inputs_present.view(1, -1), nn_inputs_past.view(1, -1))
                control_effort = torch.clamp(control_effort, -2., 2.)
                # control_effort = torch.tanh(control_effort)  # TODO
                control_force = control_effort[0, :2]
                loss_inputs['d_control_force'] = control_force - last_control_force
                loss_inputs['control_force'] = control_force
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

                # # Force 2 TODO
                # if not inp.translation_only:
                #     control_force2 = control_effort[0, 2:] * torch.as_tensor((0, 1)).cuda()  # TODO
                #     control_force_global2 = rotate(control_force2, angle_tensor)  # Control force at global reference of frame (used for visualization only)
                #     delta_control_effort = torch.cat([delta_control_effort, control_force2 - last_control_force2])

                # Save quantities necessary for loss
                # loss_inputs = update_inputs(loss_inputs, loss_inputs_present, delta_control_effort)
                loss, loss_terms = calculate_loss(loss_inputs, inp.online['hyperparams'], inp.translation_only)
                if (i - last_backprop == inp.online['n_before_backprop']):
                    # if torch.rand(1) < inp.online["simulation_dropout"]:
                    if True:  # TODO
                        loss.backward()
                        optimizer.step()
                        i_bp += 1
                        if i_bp % inp.online["model_export_stride"] == 0:
                            model_id = int(i_bp / inp.online["model_export_stride"])
                            print(f"Saving model {model_id}")
                            torch.save(model, os.path.abspath(f"{inp.export_path}/trained_model_{model_id:04d}.pth"))
                            weigths, biases = get_weights_and_biases(model)
                            for var in (weigths, biases):
                                for tag, value in var.items():
                                    writer.add_histogram(tag, value, global_step=i_bp)
                            writer.flush()
                        if i_bp == inp.online['n_iterations']: break
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
                    # last_control_force2 = last_control_force2.detach()
                    # loss_inputs_present = loss_inputs_present.detach()
                    # control_force2 = control_force.detach()
                    # control_force_global2 = control_force_global.detach()
                    sim.detach_variables()
                    last_backprop = i
            nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_effort)
            # Export values
            if (i % inp.export_stride != 0): continue
            # print(f"\n Case: {case}, i: {i} \n ")
            probes_points = probes.get_points_as_tensor()
            sim.probes_vx = sim.velocity.x.sample_at(probes_points).native().detach()
            sim.probes_vy = sim.velocity.y.sample_at(probes_points).native().detach()
            sim.probes_points = probes_points.native().detach()
            sim.loss = loss.detach()
            for key, value in loss_terms.items():
                setattr(sim, f"loss_{key}", value.detach())
            sim.reference_x = objective_xy[0, case].detach()
            sim.reference_y = objective_xy[1, case].detach()
            sim.control_force_x, sim.control_force_y = control_force_global.detach() * ref_vars['force']
            sim.error_x = loss_inputs['error_x'].detach() * ref_vars['length']
            sim.error_y = loss_inputs['error_y'].detach() * ref_vars['length']
            if not inp.translation_only:
                sim.error_x, sim.error_y = rotate([sim.error_x, sim.error_y], angle_tensor)
                sim.reference_angle = objective_ang[case].detach()
                sim.error_ang = loss_inputs['error_ang'].detach() * ref_vars['angle']
                sim.control_torque = control_torque.detach() * ref_vars['torque']
                # sim.control_force_x2, sim.control_force_y2 = control_force_global2.detach() * ref_vars['force']  # TODO
            sim.export_data(inp.export_path, case, int(i / inp.export_stride), inp.export_vars, (case == 0 and i == 0))
            # Calculate how much time is left
            current_time = time()
            steps_left = (((n_simulations - 1) - (case + 1)) * inp.online['n_timesteps'] + inp.online['n_timesteps'] - (i + 1)) / inp.export_stride
            time_left = steps_left * (current_time - last_time) / 3600
            last_time = current_time
            time_left_hours = int(time_left)
            time_left_minutes = int((time_left - time_left_hours) * 60)
            print(f"Time left: {time_left_hours:d}h {time_left_minutes:d} min")
            i += 1
        case += 1

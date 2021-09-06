from InputsManager import InputsManager
# import torch.utils.tensorboard as tb
from NeuralController import NeuralController
from misc_funcs import *
import argparse


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
    parser.add_argument("export_path", help="path that will contain run folder")
    parser.add_argument("runname", help="name of run that will be used in logs and folder")
    args = parser.parse_args()
    inp.export_path = args.export_path + args.runname + '/'
    # ----------------------------------------------
    # ---------------- Setup simulation ------------
    # ----------------------------------------------
    sim = TwoWayCouplingSimulation(inp.device, inp.translation_only)
    sim.set_initial_conditions(inp.simulation['obs_width'], inp.simulation['obs_height'], path=os.path.abspath(inp.simulation['path']))
    probes = Probes(
        inp.simulation['obs_width'] / 2 + inp.probes_offset,
        inp.simulation['obs_height'] / 2 + inp.probes_offset,
        inp.probes_size,
        inp.probes_n_rows,
        inp.probes_n_columns,
        inp.simulation['obs_xy'])
    ref_vars = dict(
        velocity=inp.simulation['inflow_velocity'],
        length=inp.simulation['obs_width'],
        force=inp.simulation['obs_mass'] * inp.max_acc,
        angle=PI,
        torque=inp.simulation['obs_inertia'] * inp.max_ang_acc,
        time=inp.simulation['obs_width'] / inp.simulation['inflow_velocity'],
        ang_velocity=inp.simulation['inflow_velocity'] / inp.simulation['obs_width']
    )
    destinations_zone_size = inp.simulation['domain_size'] - inp.online['destinations_margins'] * 2
    # ----------------------------------------------
    # ------------------- Setup NN -----------------
    # ----------------------------------------------
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
            2 if inp.translation_only else 3,
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
    prepare_export_folder(inp.export_path, first_case)  # TODO
    torch.save(model, os.path.abspath(f"{inp.export_path}/initial_model_{first_case}.pth"))
    # Create objectives
    xy = torch.rand(2, inp.online['n_cases'])
    ang = torch.rand(inp.online['n_cases'])
    # Gradually increase objectives distances
    for case in range(inp.online['n_cases']):
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
    for case in range(first_case, inp.online['n_cases']):
        # Setup case with default initial values
        sim.setup_world(
            inp.simulation['re'],
            inp.simulation['domain_size'],
            inp.simulation['dt'],
            inp.simulation['obs_mass'],
            inp.simulation['obs_inertia'],
            inp.simulation['inflow_velocity'])
        # Variables initialization
        nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).to(device)
        if inp.translation_only:
            loss_inputs = torch.zeros((inp.online['n_before_backprop'], 1, 6)).to(device)
            control_effort = torch.zeros(2).to(device)
        else:
            loss_inputs = torch.zeros((inp.online['n_before_backprop'], 1, 9)).to(device)
            control_effort = torch.zeros(3).to(device)
        last_control_force = torch.zeros(2).to(device)
        control_force_global = torch.zeros(2).to(device)
        control_torque = torch.zeros(1).to(device)
        last_control_torque = torch.zeros(1).to(device)
        loss = torch.zeros(1).to(device)
        loss_terms = torch.zeros((5)).to(device)
        # Run simulation
        for i in range(1, inp.online['n_steps'] + 1):
            sim.apply_forces(control_force_global * ref_vars['force'], control_torque * ref_vars['torque'])
            sim.advect()
            if math.any(sim.obstacle.geometry.center > inp.simulation['domain_size']) or math.any(sim.obstacle.geometry.center < (0, 0)): break  # In case obstacle escapes domain
            sim.make_incompressible()
            probes.update_transform(sim.obstacle.geometry.center.numpy(), -(sim.obstacle.geometry.angle.numpy() - math.PI / 2.0))
            sim.calculate_fluid_forces()
            # Control
            nn_inputs_present, loss_inputs_present = extract_inputs(sim, probes, objective_xy[:, case], objective_ang[case], ref_vars, inp.translation_only)
            if i < inp.past_window + 1: last_backprop = i  # Wait to backprop until past inputs are cached
            else:
                control_effort = model(nn_inputs_present, nn_inputs_past)
                control_effort = torch.clamp(control_effort, -1., 1.)
                control_force = control_effort[0, :2]
                angle_tensor = -(sim.obstacle.geometry.angle - math.PI / 2.0).native()
                control_force_global = rotate(control_force, angle_tensor)
                # Additional inputs for loss
                delta_control_effort = control_force - last_control_force
                last_control_force = control_force
                if not inp.translation_only:
                    control_torque = control_effort[0, -1:]
                    delta_control_effort = torch.cat([delta_control_effort, control_torque - last_control_torque])
                    last_control_torque = control_torque
                # Save quantities necessary for loss
                loss_inputs = update_inputs(loss_inputs, loss_inputs_present, delta_control_effort)  # TODO check if this is right
                loss, loss_terms = calculate_loss(loss_inputs, inp.online['hyperparams'], inp.translation_only)
                if (i - last_backprop == inp.online['n_before_backprop']):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # Reset graph
                    control_effort = control_effort.detach().clone()
                    last_control_force = last_control_force.detach().clone()
                    last_control_torque = last_control_torque.detach().clone()
                    loss_inputs_present = loss_inputs_present.detach().clone()
                    loss_inputs = loss_inputs.detach().clone()
                    nn_inputs_present = nn_inputs_present.detach().clone()
                    nn_inputs_past = nn_inputs_past.detach().clone()
                    control_force = control_force.detach().clone()
                    control_force_global = control_force_global.detach().clone()
                    control_torque = control_torque.detach().clone()
                    sim.detach_variables()
                    last_backprop = i
            nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_effort)
            # Export values
            if (i % inp.export_stride != 0): continue
            print(f"\n \Case: {case}, i: {i} \n ")
            probes_points = probes.get_points_as_tensor()
            sim.probes_vx = sim.velocity.x.sample_at(probes_points).native().detach()
            sim.probes_vy = sim.velocity.y.sample_at(probes_points).native().detach()
            sim.probes_points = probes_points.native().detach()
            sim.loss = loss.detach().clone()
            for key, value in loss_terms.items():
                setattr(sim, f"loss_{key}", value.detach().clone())
            sim.reference_x = objective_xy[0, case].detach().clone()
            sim.reference_y = objective_xy[1, case].detach().clone()
            sim.control_force_x, sim.control_force_y = control_force_global.detach().clone() * ref_vars['force']
            sim.error_x, sim.error_y = rotate(loss_inputs_present[0, :2].detach().clone() * ref_vars['length'], angle_tensor)
            if not inp.translation_only:
                sim.reference_angle = objective_ang[case].detach().clone()
                sim.error_ang = loss_inputs_present[0, 4].detach().clone() * ref_vars['angle']
                sim.control_torque = control_torque.detach().clone() * ref_vars['torque']
            sim.export_data(inp.export_path, case, int(i / inp.export_stride), inp.export_vars, (case == 0 and i == 0))
            torch.save(model, os.path.abspath(f"{inp.export_path}/trained_model_{case:04d}.pth"))
        lr_scheduler.step()  # Decay learning rate after every case

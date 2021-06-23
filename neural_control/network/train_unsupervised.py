from InputsManager import InputsManager
# import torch.utils.tensorboard as tb
from NeuralController import NeuralController
from misc_funcs import *


if __name__ == "__main__":
    # ----------------------------------------------
    # -------------------- Inputs ------------------
    # ----------------------------------------------
    inp = InputsManager("/home/ramos/felix/PhiFlow/neural_obstacle_control/inputs.json")
    inp.calculate_properties()
    TORCH_BACKEND.set_default_device("GPU")
    # ----------------------------------------------
    # ------------------ Setup world ---------------
    # ----------------------------------------------
    sim = TwoWayCouplingSimulation()
    i0 = sim.set_initial_conditions(inp.obs_width, inp.obs_height, path=inp.sim_load_path)
    probes = Probes(
        inp.obs_width / 2 + inp.probes_offset,
        inp.obs_height / 2 + inp.probes_offset,
        inp.probes_size,
        inp.probes_n_rows,
        inp.probes_n_columns,
        inp.obs_xy)
    ref_vars = dict(
        velocity=inp.inflow_velocity,
        length=inp.obs_width,
        force=inp.obs_mass * inp.max_acc,
        angle=PI,
        torque=inp.obs_inertia * inp.max_ang_acc,
        time=inp.obs_width / inp.inflow_velocity
    )
    # ----------------------------------------------
    # ------------------- Setup NN -----------------
    # ----------------------------------------------
    first_case = 0
    model_file = natsorted((name for name in os.listdir(inp.nn_model_path) if ".pth" in name))
    model = torch.load(inp.nn_model_path + model_file[-1])
    # model = torch.load(f"/home/ramos/work/PhiFlow2/PhiFlow/storage/model_lstm_only_translation.pth")  # TODO currently hardcoded
    model.past_window = inp.past_window  # Useful when saving model
    model.to(torch.device("cuda"))
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total amount of trainable parameters: {total_params}")
    optimizer = optim.Adam(model.parameters(), lr=inp.learning_rate_unsupervised)
    # ----------------------------------------------
    # --------------- Prepare export ---------------
    # ----------------------------------------------
    if first_case == 0:
        shutil.rmtree(f"{inp.sim_export_path}/tensorboard/", ignore_errors=True)
        prepare_export_folder(inp.sim_export_path, i0)
    # writer = tb.SummaryWriter(inp.nn_model_path + '/tensorboard/')
    # ----------------------------------------------
    # ----------------- Simulation -----------------
    # ----------------------------------------------
    for case in range(first_case, inp.n_cases):
        # Setup case with default initial values
        sim.setup_world(inp.domain_size, inp.dt, inp.obs_mass, inp.obs_inertia, inp.inflow_velocity)
        nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).cuda()
        loss_inputs = torch.zeros((inp.n_steps_before_backprop, 1, 6)).cuda()  # TODO currently hardcoded
        # loss_inputs = torch.zeros((inp.n_steps_before_backprop, 1, 9)).cuda()
        control_force = torch.zeros(2).cuda()
        control_torque = torch.zeros(1).cuda()
        loss = torch.zeros(1).cuda()
        loss_terms = torch.zeros((5)).cuda()
        x_objective = (torch.rand(2) * torch.tensor([40, 20]) + torch.tensor([40, 20])).cuda()
        ang_objective = (torch.rand(1) * 2 * PI - PI).cuda()
        integrator_force = 0
        integrator_torque = 0
        torch.cuda.empty_cache()
        # Run simulation steps
        for i in range(i0 + 1, i0 + inp.n_steps + 1):
            sim.apply_forces(control_force * ref_vars['force'], control_torque * ref_vars['torque'])
            sim.advect()
            if math.any(sim.obstacle.geometry.center > inp.domain_size) or math.any(sim.obstacle.geometry.center < (0, 0)): break  # In case obstacle escapes domain
            sim.make_incompressible()
            probes.update_transform(sim.obstacle.geometry.center.numpy(), -(sim.obstacle.geometry.angle.numpy() - math.PI / 2.0))
            sim.calculate_fluid_forces()
            # Control
            nn_inputs_present, loss_inputs_present = extract_inputs(sim, probes, x_objective, ang_objective, ref_vars)
            if i - i0 < inp.past_window + 1: last_backprop = i  # Wait until past inputs are cached
            else:
                control_effort = model(nn_inputs_past, nn_inputs_present)
                # integrator_force += 0.002 * torch.sum(loss_inputs[:, :, :2], dim=[0, 1])
                # integrator_torque -= 0.02 * torch.sum(loss_inputs[:, :, 4:5], dim=[0, 1])
                control_force = control_effort[0, :2]
                control_force = torch.clamp(control_force, -1., 1.)
                # control_torque = control_effort[0, -1:] + integrator_torque * ref_vars['torque']
                # control_torque = torch.clamp(control_torque, -1., 1.)
                # loss_inputs = update_inputs(loss_inputs, loss_inputs_present, control_force, control_torque)
                loss_inputs = update_inputs(loss_inputs, loss_inputs_present, control_force)
                loss, *loss_terms = calculate_loss(loss_inputs)
                if (i - last_backprop == inp.n_steps_before_backprop):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # Reset graph
                    # integrator_force = integrator_force.detach().clone()
                    # integrator_torque = integrator_torque.detach().clone()
                    loss_inputs_present = loss_inputs_present.detach().clone()
                    loss_inputs = loss_inputs.detach().clone()
                    nn_inputs_present = nn_inputs_present.detach().clone()
                    nn_inputs_past = nn_inputs_past.detach().clone()
                    control_force = control_force.detach().clone()
                    control_torque = control_torque.detach().clone()
                    sim.detach_variables()
                    last_backprop = i
            # nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_force, control_torque)
            nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_force)
            # Export values
            if (i % inp.export_stride != 0): continue
            print(
                f"\n \
                Case: {case}, i: {i} \n \
                Geo center: {sim.obstacle.geometry.center} \n \
                Objective: {x_objective} \n \
                Velocity: {sim.obstacle.velocity} \n  \
                Control force: {control_force} \n  \
                Fluid force: {sim.fluid_force} \n  \
                Ang Vel: {sim.obstacle.angular_velocity} \n"
            )
            probes_points = probes.get_points_as_tensor()
            sim.probes_vx = sim.velocity.x.sample_at(probes_points).native().detach()
            sim.probes_vy = sim.velocity.y.sample_at(probes_points).native().detach()
            sim.probes_points = probes_points.native().detach()
            sim.loss = loss.detach().clone()
            sim.loss_pos = loss_terms[0].detach().clone()
            sim.loss_vel = loss_terms[1].detach().clone()
            # sim.loss_ang = loss_terms[2].detach().clone()
            # sim.loss_ang_vel = loss_terms[3].detach().clone()
            sim.reference_x = x_objective[0].detach().clone()
            sim.reference_y = x_objective[1].detach().clone()
            sim.reference_angle = ang_objective.detach().clone()
            sim.error_x = loss_inputs_present[0, 0].detach().clone() * ref_vars['length']
            sim.error_y = loss_inputs_present[0, 1].detach().clone() * ref_vars['length']
            # sim.error_ang = loss_inputs_present[0, 4].detach().clone() * ref_vars['angle']
            sim.control_force_x, sim.control_force_y = control_force.detach().clone() * ref_vars['force']
            sim.control_torque = control_torque.detach().clone() * ref_vars['torque']
            sim.export_data(inp.sim_export_path, case, int(i / inp.export_stride), inp.export_vars, (case == 0 and i == 0))
            # Log scalars
            # writer.add_scalars('Loss', {f'{case}': sim.loss}, global_step=int(i / inp.export_stride))
            # Log trainable parameters
            # step = int(i / inp.export_stride) + int(inp.n_steps / inp.export_stride) * case
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         writer.add_histogram(name, param.data, global_step=step)
            # writer.flush()
            torch.save(model, f"{inp.nn_model_path}/model_{case:04d}_trained.pth")
            inp.export(f"{inp.nn_model_path}/inputs.json")

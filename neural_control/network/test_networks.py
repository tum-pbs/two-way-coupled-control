import time
from InputsManager import InputsManager
from misc_funcs import *

if __name__ == "__main__":

    models_path = ["/home/ramos/felix/PhiFlow/neural_obstacle_control/network/model_0004_trained.pth"]  # TODO currently hardcoded
    models_name = ["unsupervised"]
    sim_export_paths = ["/home/ramos/felix/PhiFlow/neural_obstacle_control/simulation_data/test/"]
    inp = InputsManager("/home/ramos/felix/PhiFlow/neural_obstacle_control/inputs.json")
    inp.calculate_properties()
    tests = InputsManager("/home/ramos/felix/PhiFlow/neural_obstacle_control/tests.json").__dict__
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
    # ----------------------------------------------
    # ------------------- Setup NN -----------------
    # ----------------------------------------------
    ref_vars = dict(
        velocity=inp.inflow_velocity,
        length=inp.obs_width,
        force=inp.obs_mass * inp.max_acc,
        angle=PI,
        torque=inp.obs_inertia * inp.max_ang_acc,
        time=inp.obs_width / inp.inflow_velocity
    )
    last = 0
    for model_path, model_name, sim_export_path in zip(models_path, models_name, sim_export_paths):
        model = torch.load(model_path).to(torch.device("cuda"))
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n Total amount of trainable parameters: {total_params}")
        # ----------------------------------------------
        # ------------------ Simulate ------------------
        # ----------------------------------------------
        with torch.no_grad():
            is_first_export = True  # Used for deleting previous files on folder
            for test_i, test_attr in enumerate(value for key, value in tests.items() if 'test' in key):
                sim.setup_world(inp.domain_size, inp.dt, inp.obs_mass, inp.obs_inertia, inp.inflow_velocity)
                nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).cuda()
                loss_inputs = torch.zeros((inp.n_steps_before_backprop, 1, 6)).cuda()  # TODO currently hardcoded
                # loss_inputs = torch.zeros((inp.n_steps_before_backprop, 1, 9)).cuda()
                control_force = torch.zeros(2).cuda()
                control_torque = torch.zeros(1).cuda()
                integrator_force = 0
                integrator_torque = 0
                for i in range(i0 + 1, i0 + test_attr['n_steps'] + 1):
                    for x_objective_, ang_objective_, objective_i in zip(test_attr['positions'], test_attr['angles'], test_attr['i']):
                        # Check if objective changed
                        if i - i0 > objective_i:
                            x_objective = torch.tensor(x_objective_).cuda()
                            ang_objective = torch.tensor(ang_objective_).cuda()
                    sim.apply_forces(control_force * ref_vars['force'], control_torque * ref_vars['torque'])
                    sim.advect()
                    sim.make_incompressible()
                    probes.update_transform(sim.obstacle.geometry.center.numpy(), -(sim.obstacle.geometry.angle.numpy() - math.PI / 2.0))
                    sim.calculate_fluid_forces()
                    # Control
                    nn_inputs_present, loss_inputs_present = extract_inputs(sim, probes, x_objective, ang_objective, ref_vars)
                    control_effort = model(nn_inputs_past, nn_inputs_present)
                    # integrator_force += 0.001 * loss_inputs_present[0, :2]
                    # integrator_torque -= 0.0005 * loss_inputs_present[0, 4:5]
                    control_force = control_effort[0, :2]  # + integrator_force
                    # control_torque = control_effort[0, -1:] + integrator_torque
                    control_force = torch.clamp(control_force, -1., 1.)
                    # control_torque = torch.clamp(control_torque, -1., 1.)
                    # loss_inputs = update_inputs(loss_inputs, loss_inputs_present, control_force, control_torque)
                    loss_inputs = update_inputs(loss_inputs, loss_inputs_present, control_force)
                    loss, *loss_terms = calculate_loss(loss_inputs)
                    # nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_force, control_torque)
                    nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_force)
                    if math.any(sim.obstacle.geometry.center > inp.domain_size) or math.any(sim.obstacle.geometry.center < (0, 0)):
                        break
                    now = time.time()
                    delta = now - last
                    i_remaining = (len(tests.keys()) - 2 - test_i) * test_attr['n_steps'] + test_attr['n_steps'] - i0
                    remaining = i_remaining * delta
                    remaining_h = np.floor(remaining / 60. / 60.)
                    remaining_m = np.floor(remaining / 60. - remaining_h * 60.)
                    last = now
                    # Export
                    if (i % inp.export_stride != 0) or (i - i0 < inp.past_window + 1):
                        continue
                    print(
                        f"\n \
                        Model: {model_name}, i: {i} \n \
                        Geo center: {sim.obstacle.geometry.center} \n \
                        Objective: {x_objective} \n \
                        Velocity: {sim.obstacle.velocity} \n  \
                        Control force: {control_force} \n  \
                        Fluid force: {sim.fluid_force} \n  \
                        Ang Vel: {sim.obstacle.angular_velocity} \n \
                        Time left: {remaining_h:.0f}h and {remaining_m:.0f} min \n"
                    )
                    if i % inp.export_stride != 0:
                        continue
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
                    # sim.reference_angle = ang_objective.detach().clone()
                    sim.error_x = loss_inputs_present[0, 0].detach().clone() * ref_vars['length']
                    sim.error_y = loss_inputs_present[0, 1].detach().clone() * ref_vars['length']
                    # sim.error_ang = loss_inputs_present[0, 4].detach().clone() * ref_vars['angle']
                    sim.control_force_x, sim.control_force_y = control_force.detach().clone() * ref_vars['force']
                    # sim.control_torque = control_torque.detach().clone() * ref_vars['torque']
                    sim.export_data(sim_export_path, test_i, int(i / inp.export_stride), inp.export_vars, is_first_export)
                    is_first_export = False

import time
from Dataset import Dataset
from InputsManager import InputsManager
from misc_funcs import *
CUDA_LAUNCH_BLOCKING = 1
torch.set_printoptions(sci_mode=True)
if __name__ == "__main__":
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json")
    inp.calculate_properties()
    tests = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json")
    models = {
        # "unsupervised_lossv2":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/unsupervised_local_coordinates_lossv2_xxx/model_0199_trained.pth",
        #     "pw": inp.past_window,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/temp/"
        # },
        # "all_loss_from_scratch":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/loss_v2/model_0199_trained.pth",
        #     "pw": inp.past_window,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/tests/loss_allterms/"
        # },
        # "lossv2":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/unsupervised_local_coordinates_lossv2_xxx/model_0199_trained.pth",
        #     "pw": inp.past_window,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage//rotation_and_translation/tests/loss_allterms_midtraining/"
        # },
        # "angle_loss":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/rotation_and_translation/unsupervised_local_coordinates_angle_loss/model_0199_trained.pth",
        #     "pw": inp.past_window,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage//rotation_and_translation/tests/more_angle_midtraining/"
        # }

        # "unsupervised_loss_decay":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/loss_v2_lrdecay/model_0199_trained.pth",
        #     "pw": 3,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests/loss_v2_decay/"
        # },
        # Supervised fc
        # "fc_pw00":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/supervised/fc/model_pw00_trained.pth",
        #     "pw": 0,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests//fc_pw00_inflow_doubled_int/"
        # },
        # "fc_pw01":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/supervised/fc/model_pw01_trained.pth",
        #     "pw": 1,
        # },
        # "fc_pw02":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/supervised/fc/model_pw02_trained.pth",
        #     "pw": 2
        # },
        # "fc_pw03":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/supervised/fc/model_pw03_trained.pth",
        #     "pw": 3
        # },
        # Supervised LSTM
        # "lstm_pw01":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/supervised/lstm/model_pw01_trained.pth",
        #     "pw": 1,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests_translation/lstm_pw01/"
        # },
        # "lstm_pw02":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/supervised/lstm/model_pw02_trained.pth",
        #     "pw": 2,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests_translation/lstm_pw02/"
        # },
        # "lstm_pw03":
        # {
        #     "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/supervised/lstm/model_pw03_trained.pth",
        #     "pw": 3,
        #     "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/tests_translation/lstm_pw03/"
        # },
        "unsupervised_translation":
        {
            "path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/unsupervised/model_0199_trained.pth",
            "pw": 3,
            "export_path": "/home/ramos/work/PhiFlow2/PhiFlow/storage/translation/tests/unsupervised_inflow_doubled_int/"
        },
    }

    # ----------------------------------------------
    # ------------------ Setup world ---------------
    # ----------------------------------------------
    sim = TwoWayCouplingSimulation(inp.translation_only)
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
        time=inp.obs_width / inp.inflow_velocity,
        ang_velocity=inp.inflow_velocity / inp.obs_width
    )
    last = 0
    dataset = Dataset(inp.supervised_datapath, inp.tvt_ratio, ref_vars)
    for model_name, model_attr in models.items():
        model_path = model_attr["path"]
        if "export_path" in model_attr.keys(): export_path = model_attr["export_path"]
        else: export_path = "/".join(model_path.split("/")[:-1]) + f"/test_{model_name}/"
        inp.past_window = model_attr["pw"]
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
            dataset.set_past_window_size(inp.past_window)
            dataset.set_mode("validation")
            is_first_export = True  # Used for deleting previous files on folder
            for test_i, test_attr in enumerate(value for key, value in tests.__dict__.items() if 'test' in key):
                sim.setup_world(inp.domain_size, inp.dt, inp.obs_mass, inp.obs_inertia, inp.inflow_velocity * 2)   # TODO XXX
                nn_inputs_past = torch.zeros((inp.past_window, 1, inp.n_past_features)).cuda()
                # n_loss = 6 if inp.translation_only else 9
                # loss_inputs = torch.zeros((inp.n_steps_before_backprop, 1, n_loss)).cuda()
                control_force = torch.zeros(2).cuda()
                control_force_global = torch.zeros(2).cuda()
                control_torque = torch.zeros(1).cuda()
                integrator_force = 0
                integrator_torque = 0
                for i in range(i0 + 1, i0 + test_attr['n_steps'] + 1):
                    for x_objective_, ang_objective_, objective_i in zip(test_attr['positions'], test_attr['angles'], test_attr['i']):
                        # Check if objective changed
                        if i - i0 > objective_i:
                            x_objective = torch.tensor(x_objective_).cuda()
                            ang_objective = torch.tensor(ang_objective_).cuda()
                    sim.apply_forces(control_force_global * ref_vars['force'], control_torque * ref_vars['torque'])
                    sim.advect()
                    sim.make_incompressible()
                    probes.update_transform(sim.obstacle.geometry.center.numpy(), -(sim.obstacle.geometry.angle.numpy() - math.PI / 2.0))
                    sim.calculate_fluid_forces()
                    # Control
                    nn_inputs_present, loss_inputs_present = extract_inputs(sim, probes, x_objective, ang_objective, ref_vars, inp.translation_only)
                    control_effort = model(nn_inputs_present, nn_inputs_past if inp.past_window else None)
                    control_effort = torch.clamp(control_effort, -1., 1.)
                    integrator_force = integrator_force + 0.002 * loss_inputs_present[0, :2]
                    control_force = control_effort[0, :2] + integrator_force
                    print(integrator_force)
                    angle_tensor = -(sim.obstacle.geometry.angle - math.PI / 2.0).native()
                    control_force_global = rotate(control_force, angle_tensor)
                    # Help stage
                    if i < inp.test_help_i and inp.translation_only:
                        if i <= inp.past_window + i0:
                            dataset_i = i - (i0 + 1)
                            _, gt_inputs_past, _, indexes = dataset.get_values_by_case_snapshot(test_i, [dataset_i])
                            control_force_global = gt_inputs_past[0][[[indexes["control_force_x"][0], indexes["control_force_y"][0]]]]
                        else:
                            dataset_i = i - (i0 + 1) - inp.past_window
                            gt_inputs_present, gt_inputs_past, gt_force, indexes = dataset.get_values_by_case_snapshot(test_i, [dataset_i])
                            control_force_global = gt_force.view(-1)
                        control_effort = control_force_global.view(-1)
                        print("Using GT values")
                    if not inp.translation_only:
                        control_torque = control_effort[0, -1:]  # + integrator_torque
                    nn_inputs_past = update_inputs(nn_inputs_past, nn_inputs_present, control_effort)
                    if math.any(sim.obstacle.geometry.center > inp.domain_size) or math.any(sim.obstacle.geometry.center < (0, 0)):
                        break
                    now = time.time()
                    delta = now - last
                    i_remaining = (len(tests.__dict__.keys()) - 2 - test_i) * test_attr['n_steps'] + test_attr['n_steps'] - i0
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
                    # sim.loss = loss.detach().clone()
                    # sim.loss_pos = loss_terms[0].detach().clone()
                    # sim.loss_vel = loss_terms[1].detach().clone()
                    sim.reference_x = x_objective[0].detach().clone()
                    sim.reference_y = x_objective[1].detach().clone()
                    sim.error_x, sim.error_y = rotate(loss_inputs_present[0, :2].detach().clone() * ref_vars['length'], angle_tensor)
                    sim.control_force_x, sim.control_force_y = control_force_global.detach().clone() * ref_vars['force']  # XXX
                    if not inp.translation_only:
                        # sim.loss_ang_vel = loss_terms[3].detach().clone()
                        # sim.loss_ang = loss_terms[2].detach().clone()
                        sim.reference_angle = ang_objective.detach().clone()
                        sim.error_ang = loss_inputs_present[0, 4].detach().clone() * ref_vars['angle']
                        sim.control_torque = control_torque.detach().clone() * ref_vars['torque']
                    sim.export_data(export_path, test_i, int(i / inp.export_stride), inp.export_vars, is_first_export)
                    is_first_export = False
            inp.export(export_path + "inputs.json")
            tests.export(export_path + "tests.json")
            torch.save(model, f"{export_path}/model{model_path.split('model')[-1]}")

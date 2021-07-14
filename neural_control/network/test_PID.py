import time
from Dataset import Dataset
from InputsManager import InputsManager
from misc_funcs import *
from PIDController import PIDController

if __name__ == "__main__":
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json")
    inp.calculate_properties(inp.translation_only)
    tests = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../tests.json").__dict__
    # ----------------------------------------------
    # ------------------ Setup world ---------------
    # ----------------------------------------------
    sim = TwoWayCouplingSimulation(inp.translation_only)
    i0 = sim.set_initial_conditions(inp.obs_width, inp.obs_height, path=inp.sim_load_path)
    # ----------------------------------------------
    # ----------------- Setup control --------------
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
    inp.Kp_xy = 1.0
    inp.Kd_xy = 15. / inp.dt
    inp.Ki_xy = 0.003
    inp.Kp_angle = 0.5
    inp.Kd_angle = 15 / inp.dt
    inp.Ki_angle = 0.003
    inp.filter_amount = 0
    controller_xy = PIDController(inp.Kp_xy, inp.Ki_xy, inp.Kd_xy, inp.filter_amount)
    controller_angle = PIDController(inp.Kp_angle, inp.Ki_angle, inp.Kd_angle, inp.filter_amount)
    # ----------------------------------------------
    # ------------------ Simulate ------------------
    # ----------------------------------------------
    with torch.no_grad():
        is_first_export = True  # Used for deleting previous files on folder
        for test_i, test_attr in enumerate(value for key, value in tests.items() if 'test' in key):
            sim.setup_world(inp.domain_size, inp.dt, inp.obs_mass, inp.obs_inertia, inp.inflow_velocity)
            control_force = torch.zeros(2).cuda()
            control_torque = torch.zeros(1).cuda()
            for i in range(i0 + 1, i0 + test_attr['n_steps'] + 1):
                for x_objective_, ang_objective_, objective_i in zip(test_attr['positions'], test_attr['angles'], test_attr['i']):
                    # Check if objective changed
                    if i - i0 > objective_i:
                        x_objective = torch.tensor(x_objective_).cuda()
                        ang_objective = torch.tensor(ang_objective_).cuda()
                sim.apply_forces(control_force * ref_vars['force'], control_torque * ref_vars['torque'])
                sim.advect()
                sim.make_incompressible()
                sim.calculate_fluid_forces()
                # Control
                error_xy = (x_objective - sim.obstacle.geometry.center).native() / ref_vars["length"]
                error_angle = (ang_objective - sim.obstacle.geometry.angle).native().view(1) / ref_vars["angle"]
                control_force = controller_xy(error_xy)
                control_force = torch.clamp(control_force, -1., 1.)
                control_torque = controller_angle(-error_angle)
                control_torque = torch.clamp(control_torque, -1., 1.)
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
                            Model: PID, i: {i} \n \
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
                sim.reference_x = x_objective[0].detach().clone()
                sim.reference_y = x_objective[1].detach().clone()
                sim.error_x, sim.error_y = error_xy.detach().clone() * ref_vars['length']
                sim.control_force_x, sim.control_force_y = control_force.detach().clone() * ref_vars['force']
                if not inp.translation_only:
                    sim.reference_angle = ang_objective.detach().clone()
                    sim.error_ang = error_angle.detach().clone()[0] * ref_vars['angle']
                    sim.control_torque = control_torque.detach().clone() * ref_vars['torque']
                sim.export_data(inp.sim_export_path, test_i, int(i / inp.export_stride), inp.export_vars, is_first_export)
                is_first_export = False
        inp.export(f"{inp.nn_model_path}/inputs.json")

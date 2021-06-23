from TwoWayCouplingSimulation import TwoWayCouplingSimulation
from InputsManager import InputsManager

if __name__ == "__main__":
    simulator = TwoWayCouplingSimulation()
    inp = InputsManager("/home/ramos/felix/PhiFlow/neural_obstacle_control/inputs.json")
    inp.calculate_properties()
    initial_i = simulator.set_initial_conditions(inp.obs_width, inp.obs_height, obs_xy=inp.obs_xy)
    simulator.setup_world(inp.domain_size, inp.dt, inp.obs_mass, inp.obs_inertia, inp.inflow_velocity)
    for i in range(initial_i, inp.n_steps):
        simulator.advect()
        simulator.make_incompressible()
        simulator.calculate_fluid_forces()
        # simulator.apply_forces()
        if i % inp.export_stride == 0:
            print(i)
            simulator.export_data(inp.sim_export_path, 0, int(i / 10), delete_previous=i == 0)
    print("Done")

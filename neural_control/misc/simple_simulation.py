import argparse
from TwoWayCouplingSimulation import TwoWayCouplingSimulation
from InputsManager import InputsManager
import os
import torch

if __name__ == "__main__":
    simulation = TwoWayCouplingSimulation("GPU")
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json", 'simulation')
    inp.calculate_properties()
    initial_i = simulation.set_initial_conditions(
        inp.simulation['obs_width'],
        inp.simulation['obs_height'],
        obs_xy=inp.simulation['obs_xy'])
    simulation.setup_world(
        inp.simulation['domain_size'],
        inp.simulation['dt'],
        inp.simulation['obs_mass'],
        inp.simulation['obs_inertia'],
        inp.simulation['inflow_velocity'])
    # Add a second box at the inflow boundary
    if inp.simulation['two_obstacles']:
        simulation.add_box(
            [inp.simulation['obs_width'], inp.simulation['domain_size'][1] / 2],
            inp.simulation['obs_width'],
            inp.simulation['obs_width'])
    with torch.no_grad():
        for i in range(initial_i, inp.simulation['n_steps']):
            simulation.advect()
            simulation.make_incompressible()
            simulation.calculate_fluid_forces()
            # simulation.apply_forces()
            if i % inp.export_stride == 0:
                print(i)
                simulation.export_data(inp.simulation['path'], 0, int(i / inp.export_stride), delete_previous=i == 0)
    inp.export(inp.simulation['path'] + "inputs.json", only=['simulation', 'probes', 'export'])
    print("Done")

import os
import numpy as np
from neural_control.InputsManager import InputsManager
from neural_control.misc.TwoWayCouplingSimulation import TwoWayCouplingSimulation
import torch
if __name__ == "__main__":
    export_vars = [
        "pressure",
        "vx",
        "vy",
        "vorticity",
        "cfl",
        "fluid_force_x",
        "fluid_force_y",
        "obs_xy",
        # "smoke",
        # "obs_vx",
        "obs2_xy",
        "obs2_ang",
        "obs2_ang_vel",
        "obs_ang"
    ]
    simulation = TwoWayCouplingSimulation("GPU")
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json", ['simulation'])
    inp.calculate_properties()
    initial_i = simulation.set_initial_conditions(
        inp.simulation['obs_type'],
        inp.simulation['obs_width'],
        inp.simulation['obs_height'],
        obs_xy=inp.simulation['obs_xy'],
    )
    obs_angle = inp.simulation.get('obs_angle', 0) / 180 * np.pi
    simulation.setup_world(
        inp.simulation['re'],
        inp.simulation['domain_size'],
        inp.simulation['dt'],
        inp.simulation['obs_mass'],
        inp.simulation['obs_inertia'],
        inp.simulation['reference_velocity'],
        inp.simulation['sponge_intensity'],
        inp.simulation['sponge_size'],
        inp.simulation['inflow_on'],
        angle=obs_angle
    )
    # Add a second box at the inflow boundary
    if inp.simulation.get('second_obstacle', False):
        params = inp.simulation['second_obstacle']
        simulation.add_box(
            params["xy"],
            params['width'],
            params['height'],
            params['ang_vel'],
        )
    with torch.no_grad():
        for i in range(initial_i, inp.simulation['n_steps']):
            simulation.advect(inp.simulation["tripping_on"])
            simulation.make_incompressible()
            simulation.calculate_fluid_forces()
            # simulation.apply_forces()
            if i % inp.export_stride == 0:
                print(f'Exporting timestep: {i}')
                print("\n")
                simulation.export_data(inp.simulation['path'], 0, int(i / inp.export_stride), delete_previous=i == 0, ids=export_vars)
    inp.export(inp.simulation['path'] + "/inputs.json", only=['simulation', 'probes', 'export'])
    print("Done")

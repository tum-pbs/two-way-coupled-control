import json
import numpy as np
from copy import deepcopy


class InputsManager():
    """
    Manager that holds inputs for simulations/network related code

    """

    def __init__(self, path: str):
        """
        Constructor loads json file and holds its values as class attribtues

        Params:
            path: path to json file

        """
        self.path = path
        with open(path, 'r') as f:
            inputs = json.load(f)
        for key, value in inputs.items():
            setattr(self, key, value)

    def calculate_properties(self):
        """
        Calculate various properties from the values from loaded json file

        """
        self.obs_mass = self.obs_density * self.obs_width * self.obs_height
        self.obs_inertia = 1 / 12.0 * (self.obs_width ** 2 + self.obs_height ** 2) * self.obs_mass  # Box's moment of inertia
        self.domain_size = np.array(self.domain_size)
        self.n_past_features = np.sum((
            (self.probes_n_rows * (self.probes_n_columns - 1)) * 2 * 4,  # Probes
            2,  # Obs velocity
            2,  # Reference xy
            2,  # Fluid forces
            2,  # Control forces
        ))
        self.n_present_features = np.sum((
            (self.probes_n_rows * (self.probes_n_columns - 1)) * 2 * 4,  # Probes
            2,  # Obs velocity
            2,  # Reference xy
            2,  # Fluid forces
        ))
        if not self.translation_only:
            self.n_past_features += np.sum((
                1,  # Reference angle
                1,  # Fluid torque
                1,  # Control torque
                1,  # Angular velocity
            ))
            self.n_present_features += np.sum((
                1,  # Reference angle
                1,  # Fluid torque
                1,  # Angular velocity
            ))

    def export(self, path: str):
        """
        Export current attributes as a json flie

        Params:
            path: path to file where values will be exported to

        """
        export_dict = deepcopy(self.__dict__)
        for key, value in export_dict.items():
            if type(value).__module__ == np.__name__:
                export_dict[key] = value.tolist()
        with open(path, 'w') as f:
            json.dump(export_dict, f, indent="    ")


if __name__ == '__main__':
    # Testing class
    inputs = InputsManager('/home/ramos/work/PhiFlow2/PhiFlow/myscripts/')
    inputs.export('/home/ramos/work/PhiFlow2/PhiFlow/myscripts/testing.json')
    print('Done')

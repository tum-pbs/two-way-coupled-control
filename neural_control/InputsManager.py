import json
import numpy as np
from copy import deepcopy


class InputsManager():
    """
    Manager that holds inputs for simulations/network related code

    """

    def __init__(self, path: str, only: list = [], exclude: list = []):
        """
        Constructor loads json file and holds its values as class attribtues

        Params:
            path: path to json file
            only: attributes to load
            exclude: do not load these attributes

        """
        with open(path, 'r') as f:
            inputs = json.load(f)
        for key, value in inputs.items():
            if only:
                if isinstance(value, dict) and not any(string in key for string in only): continue
            if exclude:
                if any(string in key for string in exclude): continue
            setattr(self, key, value)

    def calculate_properties(self):
        """
        Calculate various properties from the values from loaded json file

        """
        try:
            self.simulation['obs_mass'] = self.simulation['obs_density'] * self.simulation['obs_width'] * self.simulation['obs_height']
            self.simulation['obs_inertia'] = 1 / 12.0 * (self.simulation['obs_width'] ** 2 + self.simulation['obs_height'] ** 2) * self.simulation['obs_mass']  # Box's moment of inertia
            self.simulation['domain_size'] = np.array(self.simulation['domain_size'])
        except:
            print('Simulation properties were not calculated')
            pass
        try:
            self.online['destinations_margins'] = np.array(self.online['destinations_margins'])
        except:
            print('Online training properties were not calculated')
            pass
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

    def delete_attributes(self, string: str):
        """
        Delete attributes that have string on them

        Params:
            string: attributes with this string on their name will be deleted

        """
        keys = [key for key in self.__dict__.keys() if string in key]
        for key in keys: self.__dict__.pop(key, None)

    def export(self, path: str, exclude: list = [], only: list = []):
        """
        Export current attributes as a json flie

        Params:
            path: path to file where values will be exported to
            exclude: do not export these attributes
            only: export only attributes that have this string on their name

        """
        export_dict = deepcopy(self.__dict__)
        if only:
            entries_to_remove = [key for key in export_dict.keys() if not any(value in key for value in only)]
            for entry in entries_to_remove: export_dict.pop(entry, None)
        for key, value in export_dict.items():
            if key in exclude: continue
            if type(value).__module__ == np.__name__:
                export_dict[key] = value.tolist()
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if type(value2).__module__ == np.__name__:
                        export_dict[key][key2] = value2.tolist()
        with open(path, 'w') as f:
            json.dump(export_dict, f, indent="    ")

    def add_values(self, path: str, only: list = []):
        """
        Add values by loading a json file

        Params:
            path: path to json file
            only: if given, load only attributes that have this string on their name

        """
        with open(path, 'r') as f:
            values = json.load(f)
        for key, value in values.items():
            if only:
                if not any(string in key for string in only): continue
            if key in self.__dict__.keys():
                if value != getattr(self, key): raise ValueError(f'Trying to add existing attribute {key} with different value')
            setattr(self, key, value)


if __name__ == '__main__':
    # Testing class
    inputs = InputsManager('/home/ramos/work/PhiFlow2/PhiFlow/myscripts/')
    inputs.export('/home/ramos/work/PhiFlow2/PhiFlow/myscripts/testing.json')
    print('Done')

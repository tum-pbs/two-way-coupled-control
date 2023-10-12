import json
from collections import defaultdict
import numpy as np
import os
from copy import deepcopy
import torch
from phi.math import PI


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
                if isinstance(value, dict) and not any(string == key for string in only): continue
            if exclude:
                if any(string in key for string in exclude): continue
            setattr(self, key, value)

    def calculate_properties(self):
        """
        Calculate various properties from the values from loaded json file

        """
        try:
            if self.simulation["obs_type"] == "disc":
                area = self.simulation["obs_width"]**2 * PI
            elif self.simulation["obs_type"] == "box":
                area = self.simulation['obs_width'] * self.simulation['obs_height']
            else:
                print("Invalid obs_type")
            self.simulation['obs_mass'] = area * self.simulation['obs_density']
            # self.simulation['obs_inertia'] = 1 / 12.0 * (self.simulation['obs_width'] ** 2 + self.simulation['obs_height'] ** 2) * self.simulation['obs_mass']  # Box's moment of inertia
            self.simulation['obs_inertia'] = 4000  # To ensure numerical stability
            self.simulation['domain_size'] = np.array(self.simulation['domain_size'])
        except:
            print('Simulation properties were not calculated')
            pass
        try:
            self.unsupervised['destinations_margins'] = np.array(self.unsupervised['destinations_margins'])
        except:
            print('Unsupervised training properties were not calculated')
            pass

        n_features = defaultdict(lambda: 1)
        self.n_past_features = np.sum([n_features[var] for var in self.nn_vars])
        self.n_present_features = np.sum([n_features[var] for var in self.nn_vars if "control" not in var])

    def delete_attributes(self, keys_to_remove: list):
        """
        Delete attributes that have string on them

        Params:
            string: attributes with this string on their name will be deleted

        """
        keys = [key for key in self.__dict__.keys() if key in keys_to_remove]
        for key in keys: self.__dict__.pop(key, None)

    def export(self, filepath: str, exclude: list = [], only: list = []):
        """
        Export current attributes as a json flie

        Params:
            filepath: path to file where values will be exported to
            exclude: do not export these attributes
            only: export only attributes that have this string on their name

        """
        export_dict = deepcopy(self.__dict__)
        # Create directory in case it does not exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
        with open(os.path.abspath(filepath), 'w') as f:
            json.dump(export_dict, f, indent="    ")

    def add_values(self, path: str, only: list = []):
        """
        Add values by loading a json file

        Params:
            path: path to json file
            only: if given, load only attributes that have this string on their name

        """
        with open(os.path.abspath(path), 'r') as f:
            values = json.load(f)
        for key, value in values.items():
            if only:
                if not any(string in key for string in only): continue
            if key in self.__dict__.keys():
                if value != getattr(self, key): raise ValueError(f'Trying to add existing attribute {key} with different value')
            setattr(self, key, value)


class RLInputsManager():

    """
    This class manages the inputs that will be used for the models trained via RL

    """

    def __init__(self, past_window: int, n_features: int, n_snapshots_per_window: int, device: torch.device):
        self.past_window = past_window
        self.n_spw = n_snapshots_per_window
        self.n_features = n_features
        self._inputs = torch.zeros((1, n_features, past_window * n_snapshots_per_window)).to(device)
        self.device = device

    @property
    def values(self):
        """
        Return past inputs taking maximum absolute value of snapshots per window (conserves sign)
        Return shape should be (batch, n_features, past_window)

        """
        # Reshape so that we can take maximum absolute value easily
        _inputs = self._inputs.view(1, self.n_features, self.past_window, self.n_spw)
        indices = torch.argmax(torch.abs(_inputs), dim=3)
        inputs_max = torch.zeros(_inputs.shape[:-1]).to(self.device)
        for i in range(self.n_features):
            for j in range(self.past_window):
                inputs_max[0, i, j] = _inputs[0, i, j, indices[0, i, j]]
        return inputs_max

    def add_snapshot(self, snapshot: torch.Tensor):
        """
        Add a snapshot to the past inputs

        Params:
            snapshot: snapshot to add

        """
        assert snapshot.shape == torch.Size([1, self.n_features])
        if torch.any(self._inputs > 0):
            self._inputs = torch.cat((self._inputs[:, :, 1:], snapshot.view(1, self.n_features, 1)), dim=2)
        else:  # Initialize self._inputs
            for n in range(self._inputs.shape[-1]):
                self._inputs[..., n] = snapshot


if __name__ == '__main__':
    # Testing class
    inputs = InputsManager('/home/ramos/work/PhiFlow2/PhiFlow/myscripts/')
    inputs.export('/home/ramos/work/PhiFlow2/PhiFlow/myscripts/testing.json')
    print('Done')

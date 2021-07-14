from collections import defaultdict
import torch
import os
import numpy as np


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path: str, tvt_ratio: tuple, ref_vars=None):
        """
        Initialize ids and labels

        param: path: path to folder containing the data
        param: tvt_ratio: tuple containing the ratio of training, validation and test data
        """
        assert (np.sum(tvt_ratio) - 1.0) ** 2 < 1e-8
        self.mode = 'training'
        self.path = path + '/data/'
        self.past_window = 0
        self.vars = [
            'probes_vx',
            'probes_vy',
            'obs_vx',
            'obs_vy',
            'error_x',
            'error_y',
            'fluid_force_x',
            'fluid_force_y',
            'control_force_x',
            'control_force_y']
        self.tvt_ratio = tvt_ratio
        if ref_vars:
            self.ref_vars = ref_vars
            self.ref_vars_hash = dict(
                probes_vx='velocity',
                probes_vy='velocity',
                obs_vx='velocity',
                obs_vy='velocity',
                error_x='length',
                error_y='length',
                fluid_force_x='force',
                fluid_force_y='force',
                control_force_x='force',
                control_force_y='force'
            )
            # Make sure hash values are on ref_vars keys
            hash_values = self.ref_vars_hash.values()
            vars = list(self.ref_vars.keys())
            assert not any([value not in vars for value in hash_values])
        else:
            self.ref_vars = defaultdict(lambda: 1)
            self.ref_vars_hash = defaultdict(lambda: 1)
        self.map_cases()
        self.update()

    def __len__(self):
        """
        Denotes the total number of samples

        """
        return len(self.cases) * len(self.snapshots)

    def __getitem__(self, index, return_var_indexes: bool = False):
        """
        Generates one sample of data

        Params:
            index: index of file that will be loaded
            return_var_indexes: if True a dict with the indexes of variables is returned

        """
        case = self.cases[int(index / self.n_snapshots)]
        snapshot = self.snapshots[index % self.n_snapshots]
        var_indexes = defaultdict(list)
        i = 0
        # Past inputs
        x_past = ()  # dim = (past_window, features)
        for j in reversed(range(self.past_window)):
            for var in self.vars:
                file = f'{self.path}/{var}/{var}_case{case:04d}_{snapshot-(j+1):04d}.npy'
                data = np.load(file).reshape(-1,)
                factor = self.ref_vars[self.ref_vars_hash[var]]
                data /= factor
                x_past += (*data,)
                var_indexes[var] += [torch.arange(i, data.size + i, dtype=torch.long).view(-1)]
                i += data.size
        x_past = torch.tensor(x_past, dtype=torch.float32)
        # Present inputs
        x_present = ()
        for var in self.vars[:-2]:
            file = f'{self.path}/{var}/{var}_case{case:04d}_{snapshot:04d}.npy'
            data = np.load(file).reshape(-1,)
            factor = self.ref_vars[self.ref_vars_hash[var]]
            data /= factor
            x_present += (*data,)
        x_present = torch.tensor(x_present, dtype=torch.float32)
        # x = torch.tensor(x_past + x_present, dtype=torch.float32)
        # Labels
        y = ()
        for var in self.vars[-2:]:
            file = f'{self.path}/{var}/{var}_case{case:04d}_{snapshot:04d}.npy'
            data = np.load(file).reshape(-1)
            factor = self.ref_vars[self.ref_vars_hash[var]]
            data /= factor
            y += (*data,)
        y = torch.tensor(y, dtype=torch.float32)
        if return_var_indexes: return x_present, x_past, y, var_indexes
        else: return x_present, x_past, y

    def map_cases(self):
        """
        Map how many cases and snapshots are stored on path

        """
        directory = os.listdir(self.path)
        # Check if there are folders for all necessary variables
        for var in self.vars:
            if var not in directory:
                raise AssertionError(f'Did not found {var} folder in data path')
        self.all_cases = tuple(set([int(file.split('case')[1][:4]) for file in os.listdir(f'{self.path}/{var}') if '.npy' in file]))
        self.all_snapshots = tuple(set([int(file.split('_')[-1][:4]) for file in os.listdir(f'{self.path}/{var}') if '.npy' in file]))

    def set_mode(self, mode: str):
        """
        Set mode of dataset, e.g., training, validation or test

        param: mode: should be either "training", "validation or "test"

        """
        assert (mode == "training" or mode == "validation" or mode == "test")
        self.mode = mode
        self.update()

    def set_past_window_size(self, window_size: int):
        """
        Set how far in the past variables will be loaded

        param: window_size: how far data from previous timesteps should be loaded

        """
        assert isinstance(window_size, int)
        self.past_window = window_size

    def update(self):
        """
        Update cases and snapshots according to mode and past window

        """
        # Cases
        n_cases = len(self.all_cases)
        n_test = int(self.tvt_ratio[2] * n_cases)
        n_validation = int(self.tvt_ratio[1] * n_cases)
        if self.mode == 'test':
            self.cases = self.all_cases[:n_test]
        elif self.mode == 'validation':
            self.cases = self.all_cases[n_test: n_test + n_validation]
        elif self.mode == 'training':
            self.cases = self.all_cases[n_test + n_validation:]
        else:
            raise AssertionError('Invalid mode')
        self.n_cases = len(self.cases)
        # Snapshots
        self.snapshots = self.all_snapshots[self.past_window:]
        self.n_snapshots = len(self.snapshots)

    def get_values_by_case_snapshot(self, case: int, snapshots: list = None):
        """
        Get all labels and inputs by case

        Params:
            case: labels and inputs from this case will be returned
            snapshost: snapshots that will be loaded. If none then all snapshots from case will be loaded

        Returns:
            inputs: inputs for model
            labels: labels from case

        """
        labels_ = []
        inputs_past_ = []
        inputs_present_ = []
        assert isinstance(snapshots, list)
        if not snapshots: snapshots = range(self.n_snapshots)
        for i in snapshots:
            data = self.__getitem__(i + case * self.n_snapshots, True)
            inputs_present_ += [data[0]]
            inputs_past_ += [data[1]]
            labels_ += [data[2]]
        indexes = data[3]
        # Convert to tensor
        inputs_present = inputs_present_[0].view(1, -1)
        inputs_past = inputs_past_[0].view(1, -1)
        labels = labels_[0].view(1, -1)
        for input_present, input_past, label in zip(inputs_present_[1:], inputs_past_[1:], labels_[1:]):
            inputs_present = torch.cat((input_present, input_present.view(1, -1)))
            inputs_past = torch.cat((input_past, input_past.view(1, -1)))
            labels = torch.cat((labels, label.view(1, -1)))
        return inputs_present.cuda(), inputs_past.cuda(), labels.cuda(), indexes

    def get_destination(self, case: int):
        """
        Get destination of case

        Params:
            case: destination of this case will be returned

        Returns:
            destination: destination of case case

        """
        file_x = f'{self.path}/reference_x/reference_x_case{case:04d}_0000.npy'
        file_y = f'{self.path}/reference_y/reference_y_case{case:04d}_0000.npy'
        destination = (np.load(file_x), np.load(file_y))
        return destination


if __name__ == '__main__':
    dataset = Dataset('/home/ramos/work/PhiFlow2/PhiFlow/storage/supervised_dataset/', [0.6, 0.4, 0.])
    dataset.set_mode('training')
    dataset.set_past_window_size(3)
    dataset.update()
    params = {
        'batch_size': 10,
        'shuffle': False,
        'num_workers': 2}
    loader = torch.utils.data.DataLoader(dataset, **params)
    device = torch.device("cuda")
    import matplotlib.pyplot as plt
    for x_local, y_local in loader:
        y_local = y_local.numpy()
        break
    plt.plot(y_local[:, 0])
    y_local = []
    for i in range(10):
        y_local += [dataset[i][1].numpy()]
    y_local = np.array(y_local)
    plt.plot(y_local[:, 0])

    dataset[30]
    print(len(dataset))
    plt.show(block=True)

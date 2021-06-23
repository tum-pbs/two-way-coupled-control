import torch
import os
import numpy as np


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, path: str, tvt_ratio: tuple):
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
        self.map_cases()
        self.update()

    def __len__(self):
        """
        Denotes the total number of samples

        """
        return len(self.cases) * len(self.snapshots)

    def __getitem__(self, index):
        """
        Generates one sample of data

        param: index: index of file that will be loaded

        """
        case = self.cases[int(index / self.n_snapshots)]
        snapshot = self.snapshots[index % self.n_snapshots]
        # Past inputs
        x_past = ()  # dim = (past_window, features)
        for j in reversed(range(self.past_window)):
            for var in self.vars:
                file = f'{self.path}/{var}/{var}_case{case:04d}_{snapshot-(j+1):04d}.npy'
                data = np.load(file).reshape(-1,)
                x_past += (*data,)
        # Present inputs
        x_present = ()
        for var in self.vars[:-2]:
            file = f'{self.path}/{var}/{var}_case{case:04d}_{snapshot:04d}.npy'
            data = np.load(file).reshape(-1,)
            x_present += (*data,)
        x = torch.tensor(x_past + x_present, dtype=torch.float32)
        # Labels
        y = ()
        for i, var in enumerate(self.vars[-2:]):
            file = f'{self.path}/{var}/{var}_case{case:04d}_{snapshot:04d}.npy'
            data = np.load(file).reshape(-1)
            y += (*data,)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

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

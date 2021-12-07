from collections import defaultdict
import json
import numpy as np


class PIDController():
    """
    PID controller

    """

    def __init__(self, coefficients_files: list, filter_amount: float, clamp_values: list):
        """
        Initialization of controller

        Params:
            coefficients_files: path to jsons file with coefficients
            filter_amount: how much the error will be filtered before taking the derivative.0 = no filter and 1 will make error not change
            clamp_values: list of values to clamp the error derivative

        """
        self.n_controllers = len(coefficients_files)
        assert len(clamp_values) == self.n_controllers
        # Load coefficients from json file
        self.gains = defaultdict(list)
        for file in coefficients_files:
            with open(file, 'r') as f:
                data = json.load(f)
            self.gains['Kp'] += [float(data['Kp'])]
            self.gains['Ki'] += [float(data['Ki'])]
            self.gains['Kd'] += [float(data['Kd'])]
        self.past_error = [0 for _ in range(self.n_controllers)]
        self.integrator = [0 for _ in range(self.n_controllers)]
        self.dt = float(data['dt'])
        self.filter_amount = filter_amount
        self.clamp = clamp_values
        self.first_call = True

    def __call__(self, errors: list):
        assert len(errors) == self.n_controllers
        effort = []
        for i, error in enumerate(errors):
            error = (1 - self.filter_amount) * error + self.filter_amount * self.past_error[i]
            P = error * self.gains['Kp'][i]
            # Integral
            self.integrator[i] = (error + self.past_error[i]) / 2. * self.dt + self.integrator[i]
            I = self.gains['Ki'][i] * self.integrator[i]
            # Derivative
            derror = (error - self.past_error[i]) / self.dt
            if self.first_call: derror *= 0
            derror_clipped = [[], []]
            for j, value in enumerate(derror):
                derror_clipped[j] = np.clip(value, -self.clamp[i], self.clamp[i])
            D = self.gains['Kd'][i] * np.array(derror_clipped)
            self.past_error[i] = error
            effort += [P + D + I]
        self.first_call = False
        return effort

    def get_coeffs(self):
        return self.gains

    def export(self, path: str):
        """
        Export coefficients and dt to json file

        Params:
            path: path to export file
        """
        data = dict(self.gains)
        data['clamp'] = self.clamp
        with open(path + '/controller.json', 'w') as f:
            json.dump(data, f, indent="    ")

    def reset(self):
        self.integrator = [0 for _ in range(self.n_controllers)]
        self.past_error = [0 for _ in range(self.n_controllers)]
        self.first_call = True

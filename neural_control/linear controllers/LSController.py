import json
import numpy as np


class LSController():
    """
    Loop Shaping controller class that uses matlab generated coefficients

    """

    def __init__(self, coefficients_files: list):
        """
        Initializes controller class by loading coefficients from json files.

        Params:
            coefficients_files: paths to json files containing coefficients
        """
        self.n_controllers = len(coefficients_files)
        self.error_coeffs = []
        self.effort_coeffs = []
        self.buffers = []
        for i, coeffs_file in enumerate(coefficients_files):
            with open(coeffs_file, 'r') as f:
                data = json.load(f)
            self.error_coeffs += [[float(string) for string in data['numerator']]]
            self.effort_coeffs += [[float(string) for string in data['denominator']]]
            self.past_window = len(self.error_coeffs[i])
            buffer_size = len(self.error_coeffs[i])
            self.buffers += [{'error': [0 for _ in range(buffer_size)], 'effort': [0 for _ in range(buffer_size - 1)]}]
        self.dt = float(data['dt'])

    def __call__(self, errors: list):
        """
        Call method that calculates what the control effort should be

        Params:
            errors: error between desired and actual position for all dofs
        """
        assert len(errors) == len(self.buffers)
        effort = [0 for _ in errors]
        for i, error in enumerate(errors):
            self.buffers[i]['error'][1:] = self.buffers[i]['error'][:-1]
            self.buffers[i]['error'][0] = error
            for j in range(len(self.error_coeffs[i])):
                effort[i] += self.error_coeffs[i][j] * self.buffers[i]['error'][j]
                if j > 0:
                    effort[i] -= self.effort_coeffs[i][j] * self.buffers[i]['effort'][j - 1]
            effort[i] /= self.effort_coeffs[i][0]
            self.buffers[i]['effort'][1:] = self.buffers[i]['effort'][:-1]
            self.buffers[i]['effort'][0] = effort[i]
        return effort

    def get_coeffs(self):
        """
        Returns the coefficients of the controller as dict

        """
        return {'error': self.error_coeffs, 'effort': self.effort_coeffs}

    def export(self, path: str):
        """
        Export coefficients and dt to json file

        Params:
            path: path to export file
        """
        data = {'numerator': self.error_coeffs, 'denominator': self.effort_coeffs, 'dt': self.dt}
        with open(path + '/controller.json', 'w') as f:
            json.dump(data, f, indent="    ")

    def reset(self):
        """
        Reset buffers

        """
        for buffer in self.buffers:
            buffer['error'] = [0 for _ in range(self.past_window)]
            buffer['effort'] = [0 for _ in range(self.past_window - 1)]

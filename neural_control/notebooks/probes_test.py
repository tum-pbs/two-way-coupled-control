# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from phi.flow import *
from plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Probes():
    """
    Probe manager
    """
    def __init__(self, inner_distance : float, outter_distance : float, n_rows : int, n_columns : int, center : tuple = (0,0), rotation : float = 0):
        """
        Initialize probes

        param: inner_distance: distance to first probes row
        param: outter_distance: distance to furthest probes row
        param: n_rows: number of probes rows
        param: n_columns: number of probes columns
        """
        self.inner_distance = inner_distance
        self.width = outter_distance - inner_distance
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.center = center
        self.rotation = rotation
        self.create_probes()

    def rotate(self, angle : float, coordinates : list or tuple):
        """
        Rotate coordinates by angle

        param: angle: angle (in radians) by which coordinates will be rotated
        param: coordinates: coordinates that will be rotated
        """
        matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        rotated_coordinates = np.matmul(matrix, coordinates)
        return rotated_coordinates

    def create_probes(self):
        """
        Create probe points. First create just one row of probes then copy and rotate it until
        a whole loop os closed.
        """
        x0 = np.linspace(self.inner_distance, self.inner_distance+self.width, self.n_rows)
        strip = []
        for row in range(self.n_rows):
            x_row = np.linspace(-x0[row], x0[row], self.n_columns)
            x_row = x_row[1:]
            y_row = x0[row]
            strip += [(x,y_row) for x in x_row]
        strip = np.array(strip).transpose()
        self.__points = strip
        for angle in [np.pi/2, np.pi, -np.pi/2]:
            self.__points = np.concatenate((self.__points, self.rotate(angle, strip)), axis=1)

    def update_transform(self, new_location : list or tuple, new_rotation : float):
        """
        Update center of probes distribution

        param: new_location: new_probes_center
        """
        self.center = new_location
        self.rotation = new_rotation

    def get_points_as_tensor(self):
        """
        Get probe points with rotation and translation as a tensor
        """
        points = np.copy(self.__points)
        points = self.rotate(self.rotation, points)
        points[0,:] += self.center[0]
        points[1,:] += self.center[1]
        points = tensor(points, ('vector', 'i'))
        return points

    def get_points(self):
        """
        Get probe points with rotation and translation as a tensor
        """
        points = np.copy(self.__points)
        points = self.rotate(self.rotation, points)
        points[1,:] += self.center[1]
        points[0,:] += self.center[0]
        return points

if __name__ == '__main__':
    p = Plotter()
    grid = Domain(x=30, y=30, boundaries=OPEN, bounds=Box[0:30,0:30])
    box = Box[3:6, 3:6]
    mask = HardGeometryMask(box) >> grid.scalar_grid()
    [x,y] = np.array([values.numpy() for values in mask.points.values.unstack_spatial('x')[0].unstack('vector')])
    z = mask.values.numpy().transpose()

    probes = Probes(3, 4, 2, 3)
    probes.update_transform((4.5, 4.5), 0)
    noise = grid.scalar_grid(Noise())
    probes_values = noise.sample_at(probes.get_points_as_tensor())
    print(*probes_values)
    print(*probes.get_points())
    fig, ax = plt.subplots(1)
    # ax.scatter(x,y)
    mask_points = mask.points.numpy()
    mask_values = mask.values.numpy()
    ax.scatter(x,y,10*(z>0.5),'r')
    ax.scatter(*probes.get_points(), 20, color='k', marker='x')
    ax.imshow(noise.values.numpy().transpose(), origin='lower', extent=[mask_points.min()-0.5, mask_points.max()+0.5, mask_points.min()-0.5, mask_points.max()+0.5])
    # ax.imshow(mask_values.transpose(), origin='lower', extent=[mask_points.min()-0.5, mask_points.max()+0.5, mask_points.min()-0.5, mask_points.max()+0.5])
    plt.show(block=True)



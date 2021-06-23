# MAKE SURE THE SAME BACKEND FROM THE SIMULATION IS BEING USED (TORCH, TENSORFLOW, NUMPY OR JAX)
from phi.torch.flow import *
import numpy as np


class Probes():
    """
    This class creates and maintains an array of probes distributed in a rectangular way

    """

    def __init__(self, width_inner: float, height_inner: float, size: float, n_rows: int, n_columns: int, center: tuple = (0, 0), rotation: float = 0):
        """
        Initialize probes

        Params:
            width_inner: horizontal distance of the box's vertical wall to closest probe
            height_inner: vertical distance of the box's horizontal wall to closest probe
            size: distance between innermost and outtermost probes
            n_rows: number of probes rows (if looking at top most rows)
            n_columns: number of probes columns (if looking at top most rows)

        """
        self.width_inner = width_inner
        self.height_inner = height_inner
        self.size = size
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.center = center
        self.rotation = rotation
        self.create_probes()
        self.update_transform(center, rotation)

    def rotate(self, angle: float, coordinates: list or tuple) -> np.array:
        """
        Rotate coordinates by angle

        Params:
            angle: angle (in radians) by which coordinates will be rotated
            coordinates: coordinates that will be rotated

        Returns:
            rotated_coordinates: probes points rotated by angle

        """
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_coordinates = np.matmul(matrix, coordinates)
        return rotated_coordinates

    def create_probes(self):
        """
        Create probe points. First create just one row of probes then copy and rotate it until
        a whole loop is closed.

        """
        y0 = np.linspace(self.height_inner, self.height_inner + self.size, self.n_rows)
        x0 = np.linspace(self.width_inner, self.width_inner + self.size, self.n_rows)
        strip_top = []
        for row in range(self.n_rows):
            x_row = np.linspace(-x0[row], x0[row], self.n_columns)
            x_row = x_row[1:]
            y_row = y0[row]
            strip_top += [(x, y_row) for x in x_row]
        strip_top = np.array(strip_top).transpose()
        y0 = np.linspace(self.height_inner, self.height_inner + self.size, self.n_rows)
        x0 = np.linspace(self.width_inner, self.width_inner + self.size, self.n_rows)
        strip_right = []
        for row in range(self.n_rows):
            y_row = np.linspace(y0[row], -y0[row], self.n_columns)
            y_row = y_row[1:]
            x_row = x0[row]
            strip_right += [(x_row, y) for y in y_row]
        strip_right = np.array(strip_right).transpose()
        self.__points = np.concatenate((strip_top, strip_right), axis=1)
        for angle in [np.pi]:
            self.__points = np.concatenate((self.__points, self.rotate(angle, self.__points)), axis=1)

    def update_transform(self, new_location: list or tuple, new_rotation: float):
        """
        Update center and angle of probes distribution

        Params:
            new_location: new probes center
            new_rotation: new probes rotation

        """
        self.center = new_location
        self.rotation = new_rotation

    def get_points_as_tensor(self):
        """
        Get probe points with rotation and translation as a tensor

        Returns:
            points: coordinates of probes (tensor)

        """
        points = self.get_points()
        points = tensor(points, ('vector', 'index'))
        return points

    def get_points(self):
        """
        Get probe points with rotation and translation

        Returns:
            points: coordinates of probes

        """
        points = np.copy(self.__points)
        points = self.rotate(self.rotation, points)
        points[1, :] += self.center[1]
        points[0, :] += self.center[0]
        return points


if __name__ == '__main__':
    # Test
    import matplotlib.pyplot as plt
    grid = Domain(x=30, y=30, boundaries=OPEN, bounds=Box[0:30, 0:30])
    box = Box[10:25, 15:25]
    mask = HardGeometryMask(box) >> grid.scalar_grid()
    [x, y] = np.array([values.numpy() for values in mask.points.values.unstack_spatial('x')[0].unstack('vector')])
    z = mask.values.numpy().transpose()
    probes = Probes(box.half_size.numpy()[0] + 1, box.half_size.numpy()[1] + 1, 2, 3, 4, box.center.numpy())
    noise = grid.scalar_grid(Noise())
    probes_values = noise.sample_at(probes.get_points_as_tensor())
    print(*probes_values)
    print(*probes.get_points())
    fig, ax = plt.subplots(1)
    # ax.scatter(x,y)
    mask_points = mask.points.numpy()
    mask_values = mask.values.numpy()
    ax.scatter(x.transpose(), y.transpose(), 10 * (z > 0.5), 'r')
    ax.scatter(*probes.get_points(), 20, color='k', marker='x')
    ax.imshow(noise.values.numpy().transpose(), origin='lower', extent=[
              mask_points.min() - 0.5, mask_points.max() + 0.5, mask_points.min() - 0.5, mask_points.max() + 0.5])
    # ax.imshow(mask_values.transpose(), origin='lower', extent=[mask_points.min()-0.5, mask_points.max()+0.5, mask_points.min()-0.5, mask_points.max()+0.5])
    plt.show(block=True)

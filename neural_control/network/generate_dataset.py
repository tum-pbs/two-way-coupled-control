# import plotter; import importlib; importlib.reload(plotter); from plotter import Plotter

# from demos.myscripts.dbghelpers import plot
import dbghelpers as dbg
from Plotter import Plotter
import shutil
from shutil import copyfile

# from bspline3_interpolate import bspline3_interpolate
from natsort import natsorted
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
# from weak_coupling import *
from TwoWayCouplingSimulation import *
from InputsManager import InputsManager
import time
from Probes import Probes


class Generator(TwoWayCouplingSimulation):
    def __init__(self):
        """
        Initialize internal variables

        """
        super().__init__()
        self.n_vel_spline_knots = []
        self.initial_world = []
        self.plotter = Plotter(
            imshow_kwargs={"origin": "lower", "cmap": "seismic", "vmin": -1, "vmax": 1, "interpolation": "None"},
            plot_kwargs={"linestyle": "--", "alpha": 0.5, "linewidth": 3})

    def sample_random_curve(self, n: int, amplitude: float, seed: int):
        """
        Sample a value from a random curve created with fourier series

        param: n: number of random points that will be generated
        param: amplitude: maximum amplitude between two consecutive points
        param: seed: seed for random number generator
        """
        randomGenerator = np.random.RandomState()
        randomGenerator.seed(seed)
        y = [0.0]
        for i in range(n - 1):
            y.append(randomGenerator.uniform(0, amplitude))
        return y

    def create_trajectories(self, n_timesteps: int, destinations: list, initial_position: list, dt):
        """
        Create random knots of a spline

        param: n_timesteps: number of simulations' timesteps
        param: destinations: last points of trajectories
        param: initial_position: departure point of trajectory
        """
        velocities = []
        i = 0
        seed = 0
        while True:
            # for i in range(len(destinations)):
            knots_y_vy = [1] * 4
            knots_y_vy[0] = 0
            knots_y_vy[-1] = 0
            knots_y_vx = knots_y_vy
            # knots_y_vx = self.sample_random_curve(self.n_vel_spline_knots, 5.0, int(2 * seed) + 1)
            seed += 1
            # knots_y = [[knot_y_vy, knot_y_vx] for knot_y_vy, knot_y_vx in zip(knots_y_vy, knots_y_vx)]
            # knots_vx = [knot[0] for knot in knots]
            # knots_vy = [knot[1] for knot in knots]
            x = np.arange(len(knots_y_vy))
            cs_x = CubicSpline(x, knots_y_vy, bc_type="clamped")
            cs_y = CubicSpline(x, knots_y_vx, bc_type="clamped")
            xs = np.linspace(0, x[-1], n_timesteps)
            velocity = np.array([[vx, vy] for vx, vy in zip(cs_x(xs), cs_y(xs))])

            def integrate(velocity, n, initial_position):
                pos = [initial_position]
                for j in range(n):
                    pos += [pos[j] + velocity[j] * dt]
                pos = np.array(pos)
                return pos

            pos = integrate(velocity, n_timesteps, initial_position)
            last_pos = [pos[-1, 0], pos[-1, 1]]
            true_last_pos = destinations[i]
            velocity[:, 0] *= (true_last_pos[0] - initial_position[0]) / (last_pos[0] - initial_position[0])
            velocity[:, 1] *= (true_last_pos[1] - initial_position[1]) / (last_pos[1] - initial_position[1])
            if np.any(velocity > 8):
                i += 1
                continue  # TODO CFL condition is hardcoded
            velocities += [velocity]
            pos = integrate(velocity, n_timesteps, initial_position)
            plt.figure(1)
            plt.plot(pos[:, 0], pos[:, 1])
            plt.plot(*destinations[i], "o")
            plt.figure(2)
            plt.plot(velocity[:, 0])
            plt.plot(velocity[:, 1])
            print(i)
            i += 1
            if i >= len(destinations):
                break
        velocities = np.array(velocities)
        # plt.figure(3)
        # plt.title('Density of vx and vy for each timestep')
        # for vx in velocities[...,0].transpose():
        #     sns.kdeplot(vx,bw=0.5,color='b', alpha=0.5)
        # plt.figure(4)
        # plt.title('Density of vy for each timestep')
        # for vy in velocities[...,1].transpose():
        #     sns.kdeplot(vy,bw=0.5,color='r', alpha=0.5)
        plt.show(block=True)
        return velocities

    def create_destinations(self, n: int, bounds: list):
        """
        Create end points of trajectories

        param: n: number of trajectories
        param: bounds: list containing boundaries that endpoints should stay within
        """
        randomGenerator = np.random.RandomState()
        randomGenerator.seed(0)
        points = np.array([])
        values_x = []
        values_y = []
        for _ in range(n):
            values_x += [randomGenerator.uniform(bounds[0][0], bounds[1][0])]
            values_y += [randomGenerator.uniform(bounds[0][1], bounds[1][1])]
        points = []
        for x, y in zip(values_x, values_y):
            points += [[x, y]]
        return points
        for point in points:
            plt.plot(*point, 'o')

    def distribute_data(self, training_percentage: float, test_percentage: float = None):
        """
        Distribute data between training, test and validation. There will always be a training
        and test folder. Validation is optional.

        param: training_percentage: percentage of files that will be allocated to training
        param: test_percentage: percentage of files that will be allocated to test
        """
        files = os.listdir(self.path)
        files = [file for file in files if file.startswith("case")]
        files = natsorted(files, key=lambda x: x.lower())
        n_files = len(files)
        n_timesteps = int(files[-1].split("_")[1].split(".")[0]) + 1
        if test_percentage is None:
            n_test = 0
        else:
            n_test = int(test_percentage * n_files)
        n_test = n_test - n_test % n_timesteps  # Make sure whole simulation is in dataset
        n_training = int(training_percentage * n_files)
        n_training = n_training - n_training % n_timesteps
        n_validation = n_files - n_training - n_test
        assert n_validation + n_training + n_test == n_files
        for i in range(n_training):
            copyfile(self.path + files[i], self.path + "/training/" + files[i])
        for i in range(n_test):
            copyfile(self.path + files[i + n_training], self.path + "/test/" + files[i + n_training])
        for i in range(n_validation):
            copyfile(self.path + files[i + n_training + n_test], self.path + "/validation/" + files[i + n_training + n_test])
        for file in files:
            os.remove(self.path + file)
        print("\n %d files for training (%.2f%%) \n" % (n_training, float(n_training) / n_files * 100))
        print(" %d files for validation (%.2f%%) \n" % (n_validation, float(n_validation) / n_files * 100))
        print(" %d files for test (%.2f%%) \n" % (n_test, float(n_test) / n_files * 100))

    def set_obstacle_velocity(self, velocity):
        """
        Set velocity of obstacle

        param: velocity: velocity that will be imposed on obstacle

        """
        self.obstacle = self.obstacle.copied_with(velocity=tensor(velocity))


if __name__ == "__main__":
    export_vars = (
        "pressure",
        "vx",
        "vy",
        "obs_mask",
        "obs_xy",
        "obs_vx",
        "obs_vy",
        "obs_ang",
        "obs_ang_vel",
        "control_force_x",
        "control_force_y",
        "fluid_force_x",
        "fluid_force_y",
        "probes_vx",
        "probes_vy",
        "probes_points",
        "reference_x",
        "reference_y",
        "error_x",
        "error_y"
    )
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json")
    inp.calculate_properties()
    generator = Generator()
    generator.set_initial_conditions(inp.obs_width, inp.obs_height, inp.sim_load_path)
    # Generate trajectories
    destinations = generator.create_destinations(50, [0.4 * inp.domain_size, 0.6 * inp.domain_size])
    velocities = generator.create_trajectories(int(inp.n_steps / 2 + 1), destinations, (inp.obs_xy[0], inp.obs_xy[1]), inp.dt)
    velocities = np.concatenate((velocities, velocities * 0), axis=1)
    # Probes
    probes = Probes(inp.obs_width / 2 + inp.probes_offset, inp.obs_height / 2 + inp.probes_offset, inp.probes_size, inp.probes_n_rows, inp.probes_n_columns, inp.obs_xy)
    current = time.time()
    for i, velocity_curve in enumerate(velocities):
        generator.setup_world(inp.domain_size, inp.dt, inp.obs_mass, inp.obs_inertia, inp.inflow_velocity)
        for j in range(inp.n_steps - 1):
            generator.set_obstacle_velocity(velocity_curve[j])
            generator.advect()
            generator.make_incompressible()
            generator.calculate_fluid_forces()
            (generator.control_force_x, generator.control_force_y) = -generator.fluid_force + inp.obs_mass * (velocity_curve[j + 1] - velocity_curve[j]) / inp.dt
            # Probes
            probes.update_transform(generator.obstacle.geometry.center.numpy(), generator.obstacle.geometry.angle.numpy() - PI / 2)
            probes_points = probes.get_points_as_tensor()
            generator.probes_vx = generator.velocity.x.sample_at(probes_points).native()
            generator.probes_vy = generator.velocity.y.sample_at(probes_points).native()
            generator.probes_points = probes_points.native()
            # Reference
            generator.reference_x = destinations[i][0]
            generator.reference_y = destinations[i][1]
            generator.error_x = destinations[i][0] - generator.obstacle.geometry.center.numpy()[0]
            generator.error_y = destinations[i][1] - generator.obstacle.geometry.center.numpy()[1]
            generator.export_data(inp.supervised_datapath, i, j, export_vars, delete_previous=(i == 0 and j == 0))
            if j % 50 == 0:
                remaining = (time.time() - current) * ((inp.n_steps - 1 - j) + (len(velocities) - (i + 1)) * inp.n_steps) / 50
                remaining_h = np.floor(remaining / 60. / 60.)
                remaining_m = np.floor(remaining / 60. - remaining_h * 60.)
                current = time.time()
                print(f'{j} step of {i} case')
                print(" Time remaining: %dh %dmin" % (remaining_h, remaining_m))
    print('done')

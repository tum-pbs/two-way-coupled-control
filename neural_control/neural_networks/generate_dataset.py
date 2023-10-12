
from neural_control.visualization.Plotter import Plotter
from shutil import copyfile
from natsort import natsorted
from numpy import pi
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from neural_control.misc.TwoWayCouplingSimulation import *
from neural_control.InputsManager import InputsManager
import time
from neural_control.misc.misc_funcs import rotate


class Generator(TwoWayCouplingSimulation):
    def __init__(self, device):
        """
        Initialize internal variables

        """
        super().__init__(device)
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
            knots_y_vy = [0, 6, 4, 1, 0]
            knots_y_vx = knots_y_vy
            seed += 1
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
            plt.title('Velocity')
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
        # plt.show(block=True)
        return velocities

    def create_objective_xy(self, n: int, domain_size: list, margins: list):
        """
        Create end points of trajectories

        Params:
            n: number of trajectories
            domain_size: size of the domain
            margins: margins of the domain

        """
        randomGenerator = np.random.RandomState()
        randomGenerator.seed(0)
        points = np.array([])
        values_x = []
        values_y = []
        max_x = domain_size[0] - margins[0]
        max_y = domain_size[1] - margins[1]
        for _ in range(n):
            values_x += [randomGenerator.uniform(margins[0], max_x)]
            values_y += [randomGenerator.uniform(margins[1], max_y)]
        points = []
        for x, y in zip(values_x, values_y):
            points += [[x, y]]
        for point in points:
            plt.plot(*point, 'o')
        # plt.show(block=True)
        return points

    def create_objective_angle(self, n: int):
        """
        Create end points of trajectories

        Params:
            n: number of trajectories
            domain_size: size of the domain
            margins: margins of the domain

        """
        randomGenerator = np.random.RandomState()
        randomGenerator.seed(1)
        objectives = []
        for _ in range(n):
            objectives += [randomGenerator.uniform(-pi, pi)]
        plt.hist(objectives, bins=10)
        # plt.show(block=True)
        return objectives

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

    def set_obstacle_velocity(self, velocity, angular_velocity=0):
        """
        Set velocity of obstacle

        param: velocity: velocity that will be imposed on obstacle

        """
        self.obstacle = self.obstacle.copied_with(velocity=tensor(velocity), angular_velocity=tensor(angular_velocity))


if __name__ == "__main__":
    export_vars = (
        # "pressure",
        # "vx",
        # "vy",
        "obs_mask",
        "obs_xy",
        "obs_vx",
        "obs_vy",
        "control_force_x",
        "control_force_y",
        "fluid_force_x",
        "fluid_force_y",
        "reference_x",
        "reference_y",
        "error_x",
        "error_y",
        "control_torque",
        "fluid_torque",
        "error_ang",
        "obs_ang_vel",
        "control_force_x_local",
        "control_force_y_local",
        "obs_vx_local",
        "obs_vy_local",
        "error_x_local",
        "error_y_local",
        # "obs_ang",
        # "obs_ang_vel",
    )
    inp = InputsManager(os.path.dirname(os.path.abspath(__file__)) + "/../inputs.json", ["supervised"])
    inp.add_values(inp.supervised["initial_conditions_path"] + "inputs.json", ["probes", "simulation"])
    gen = Generator(inp.device)
    gen.set_initial_conditions(
        inp.simulation['obs_type'],
        inp.simulation['obs_width'],
        inp.simulation['obs_height'],
        inp.supervised["initial_conditions_path"]
    )
    # Generate trajectories xy
    destinations = gen.create_objective_xy(inp.supervised["n_simulations"], inp.simulation['domain_size'], inp.supervised['destinations_margins'])
    velocities = gen.create_trajectories(
        int(inp.supervised['dataset_n_steps'] / 2 + 1),
        destinations,
        (*inp.simulation['obs_xy'],),
        inp.simulation['dt'])
    shape_concat = list(velocities.shape)
    shape_concat[1] = inp.supervised['dataset_n_steps'] - shape_concat[1]
    velocities = np.concatenate((velocities, np.zeros(shape_concat)), axis=1)
    inp.supervised["destinations"] = destinations
    objective_angles = velocities[:, :, 0:1] * 0
    # Generate trajectories angle
    objective_angles = gen.create_objective_angle(inp.supervised["n_simulations"])
    # Duplicate in order to use the same functions used for generating velocities
    objective_angles2 = [[-value, -value] for value in objective_angles]  # I need to invert the sign because of phiflow left hand convetion
    ang_velocities = gen.create_trajectories(
        int(inp.supervised['dataset_n_steps'] / 2 + 1),
        objective_angles2,
        (0, 0),
        inp.simulation['dt'])
    shape_concat = list(ang_velocities.shape)
    shape_concat[1] = inp.supervised['dataset_n_steps'] - shape_concat[1]
    ang_velocities = np.concatenate((ang_velocities, np.zeros(shape_concat)), axis=1)
    ang_velocities = ang_velocities[:, :, 0]
    inp.supervised['objective_angles'] = objective_angles
    if inp.translation_only: ang_velocities *= 0
    inp.export(inp.supervised["dataset_path"] + "inputs.json")
    current = time.time()
    for i, (velocity, ang_velocity) in enumerate(zip(velocities, ang_velocities)):
        gen.setup_world(
            inp.simulation['re'],
            inp.simulation['domain_size'],
            inp.simulation['dt'],
            inp.simulation['obs_mass'],
            inp.simulation['obs_inertia'],
            inp.simulation['reference_velocity'],
            inp.simulation['sponge_intensity'],
            inp.simulation['sponge_size'],
            inp.simulation['inflow_on'])
        for j in range(velocities.shape[1] - 1):
            gen.set_obstacle_velocity(velocity[j], ang_velocity[j])
            gen.advect()
            gen.make_incompressible()
            gen.calculate_fluid_forces()
            # Add variables to generator object
            angle = (gen.obstacle.geometry.angle - math.PI / 2.0).native()  # Negative angle
            gen.error_ang = objective_angles[i] - angle
            gen.control_torque = -gen.fluid_torque + inp.simulation['obs_inertia'] * (ang_velocity[j + 1] - ang_velocity[j]) / inp.simulation['dt']
            gen.control_torque = gen.control_torque.native()[0]
            # Rotate xy vars so they are in local cooridnates
            control_force_x, control_force_y = -gen.fluid_force + inp.simulation['obs_mass'] * (velocity[j + 1] - velocity[j]) / inp.simulation['dt']
            gen.control_force_x_local, gen.control_force_y_local = rotate(torch.stack([control_force_x, control_force_y]), angle)
            gen.control_force_x, gen.control_force_y = control_force_x, control_force_y
            gen.reference_x, gen.reference_y = destinations[i][0], destinations[i][1]
            error = torch.as_tensor(destinations[i]).to(gen.device) - gen.obstacle.geometry.center.native()
            gen.error_x_local, gen.error_y_local = rotate(error, angle)
            gen.error_x, gen.error_y = error
            gen.obs_vx_local, gen.obs_vy_local = rotate(gen.obstacle.velocity.native(), angle)
            gen.export_data(inp.supervised["dataset_path"], i, j, export_vars, delete_previous=(i == 0 and j == 0))
            if j % 50 == 0:
                remaining = (time.time() - current) * ((velocities.shape[1] - 1 - j) + (len(velocities) - (i + 1)) * velocities.shape[1]) / 50
                remaining_h = np.floor(remaining / 60. / 60.)
                remaining_m = np.floor(remaining / 60. - remaining_h * 60.)
                current = time.time()
                print(f'{j} step of case {i}')
                print(" Time remaining: %dh %dmin" % (remaining_h, remaining_m))
    print('done')

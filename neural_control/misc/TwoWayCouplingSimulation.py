from collections import defaultdict
from phi.torch.flow import *
import os
from natsort import natsorted
import shutil
import copy
from phi.field import spatial_gradient


class TwoWayCouplingSimulation:
    """
    This class manages a simulation with two way coupling with a box obstacle on it. The steps are basically:
        .set_initial_conditions()
        .setup_world()
        loop:
            .apply_forces()
            .advect()
            .make_incompressible()
            .calculate_fluid_forces()

    """

    def __init__(self, device: str, translation_only: bool = False):
        """
        Class initializer. If translation_only is True, simulation won't have rotation

        Params:
            device: GPU or CPU
            translation_only: if True then obstacle will not rotate

        """
        self.ic = {}
        self.translation_only = translation_only
        self.additional_obs = []
        if device == "GPU":
            TORCH_BACKEND.set_default_device("GPU")
            self.device = torch.device("cuda:0")
        else:
            TORCH_BACKEND.set_default_device("CPU")
            self.device = torch.device("cpu")

    def set_initial_conditions(
        self,
        obs_w: float,
        obs_h: float,
        path: str = None,
        obs_xy: list = None,
        obs_ang: float = PI / 2,
        obs_vel: list = [0, 0],
        obs_ang_vel: float = 0,
    ):
        """
        Set the initial conditions of simulation. If a path is given then the initial conditions will be loaded from
        this directory. The last snapshot of last case will be used.
        Otherwise initial conditions for obstacle have to be provided.

        Params:
            obs_w: width of obstacle
            obs_h: height of obstacle
            path: path to directory that contains folders with data that can be loaded
            obs_xy: initial position of obstacle
            obs_ang: initial angle of obstacle
            obs_vel: initial velocity of obstacle
            obs_ang_vel: initial angular velocity of obstacle

        Returns:
            i0: snapshot index of loaded simulation

        """
        assert obs_xy or path  # At least one of these must be provided
        self.ic["obs_w"] = obs_w
        self.ic["obs_h"] = obs_h
        if path:
            # Get snapshot and case for loading
            files = [file for file in os.listdir(
                f"{path}/data/") if "." not in file]
            files = natsorted([file for file in os.listdir(
                f"{path}/data/{files[0]}/") if ".npy" in file])
            case_snapshot = "_".join(files[-1].split("_")[-2:])
            for var in [
                "pressure",
                "vx",
                "vy",
                "obs_xy",
                "obs_vx",
                "obs_vy",
                "obs_ang",
                "obs_ang_vel",
            ]:
                self.ic[var] = np.load(
                    f"{path}/data/{var}/{var}_{case_snapshot}")
                if var in ["vx", "vy"]:
                    self.ic[var] = math.tensor(
                        self.ic[var], ("x", "y"))
                if var == 'obs_ang': self.ic[var] += PI / 2  # Account for difference in angle convention
            # Try to load a second obstacle
            try:
                self.ic['obs2_xy'] = np.load(f"{path}/data/obs2_xy/obs2_xy_case0000_0000.npy")
            except:
                print("Did not found data of second obstacle")
                pass
            i0 = int(case_snapshot.split('_')[1][:4])
            return i0
        else:
            obs_vel = math.tensor(obs_vel)
            obs_ang = math.tensor((obs_ang,), convert=True)
            obs_ang_vel = math.tensor((obs_ang_vel,), convert=True)
            self.ic["pressure"] = 0
            self.ic["vx"] = self.ic["vy"] = math.tensor(0)
            self.ic["obs_xy"] = obs_xy
            self.ic["obs_vx"] = obs_vel[0]
            self.ic["obs_vy"] = obs_vel[1]
            self.ic["obs_ang_vel"] = obs_ang_vel
            self.ic["obs_ang"] = obs_ang
            return 0

    def setup_world(self, domain_size: list, dt: float, obs_mass: float, obs_inertia: float, inflow_velocity: float):
        """
        Setup world for simulation with a box on it

        Params:
            domain_size: size of simulation domain
            dt: step size for time marching
            obs_mass: obstacle's mass
            obs_inertia: obstacle's moment of inertia
            inflow_velocity: velocity of inflow

        """
        self.dt = dt
        self.obs_mass = obs_mass
        self.obs_inertia = obs_inertia
        self.fluid_force = math.tensor(torch.zeros(2).to(self.device))
        self.fluid_torque = math.tensor(torch.zeros(1).to(self.device))
        self.solve_params = math.LinearSolve(absolute_tolerance=1e-3, max_iterations=10e3)
        constant_velocity_bc_left = {
            "accessible_extrapolation": extrapolation.ConstantExtrapolation(1),
            "active_extrapolation": extrapolation.ConstantExtrapolation(0),
            "near_vector_extrapolation": extrapolation.ConstantExtrapolation(0),
            "scalar_extrapolation": extrapolation.ConstantExtrapolation(0),
            "vector_extrapolation": extrapolation.ConstantExtrapolation(inflow_velocity),
        }
        self.domain = Domain(
            x=domain_size[0],
            y=domain_size[1],
            boundaries=((constant_velocity_bc_left, OPEN), (OPEN, OPEN)),
            # boundaries=((OPEN, OPEN), (OPEN, OPEN)),
            bounds=Box[0: domain_size[0], 0: domain_size[1]],
        )
        self.pressure = self.domain.scalar_grid(self.ic["pressure"])
        self.velocity = self.domain.staggered_grid(
            math.channel_stack((self.ic["vx"], self.ic["vy"]), "vector"))
        # Obstacle
        box_coordinates = [
            slice(
                self.ic["obs_xy"][0] - self.ic["obs_w"] / 2,
                self.ic["obs_xy"][0] + self.ic["obs_w"] / 2,
            ),
            slice(
                self.ic["obs_xy"][1] - self.ic["obs_h"] / 2,
                self.ic["obs_xy"][1] + self.ic["obs_h"] / 2,
            ),
        ]
        obstacle_geometry = Box[box_coordinates].rotated(self.ic["obs_ang"][0])
        self.obstacle = Obstacle(
            obstacle_geometry,
            velocity=math.tensor((self.ic["obs_vx"], self.ic["obs_vy"])),
            angular_velocity=math.tensor(self.ic["obs_ang_vel"][0], convert=True),
        )
        # Add additional obstacle if it was provided
        if "obs2_xy" in self.ic.keys():
            self.add_box(self.ic["obs2_xy"], self.ic["obs_w"], self.ic["obs_w"])
        # Constant velocity
        self.inflow_velocity_mask = HardGeometryMask(Box[:0.5, :]) >> self.velocity
        self.inflow_velocity = inflow_velocity

    def advect(self):
        """
        Perform advection step of fluid and obstacle

        """
        self.velocity = advect.semi_lagrangian(self.velocity, self.velocity, self.dt)
        self.velocity = self.velocity * (1 - self.inflow_velocity_mask.values) + self.inflow_velocity_mask.values * (self.inflow_velocity, 0)
        new_geometry = self.obstacle.geometry.rotated(-self.obstacle.angular_velocity * self.dt)
        new_geometry = new_geometry.shifted(self.obstacle.velocity * self.dt)
        self.obstacle = self.obstacle.copied_with(geometry=new_geometry, age=self.obstacle.age + self.dt)

    def make_incompressible(self):
        """
        Ensure divergence free condition on velocity field

        """
        new_velocity, new_pressure, *_ = fluid.make_incompressible(
            self.velocity, self.domain, (self.obstacle, *self.additional_obs),
            solve_params=self.solve_params,
            pressure_guess=self.pressure)
        self.pressure = new_pressure
        self.velocity = new_velocity

    def add_sphere(self, xy: torch.Tensor, radius: torch.Tensor):
        """
        Add an sphere to the simulation at xy with radius r

        Params:
            xy: location of sphere
            radius: radius of sphere

        """
        self.additional_obs = (Obstacle(Sphere(torch.as_tensor(xy), torch.as_tensor(radius))),)

    def add_box(self, xy: torch.Tensor, width: torch.Tensor, height: torch.Tensor):
        """
        Add a box to the simulation at xy with width and height

        Params:
            xy: location of box
            width: width of box
            height: height of box

        """
        box_coordinates = [slice(xy[0] - width / 2, xy[0] + width / 2,),
                           slice(xy[1] - height / 2, xy[1] + height / 2,), ]
        geometry = Box[box_coordinates].shifted(torch.tensor(0).to(self.device))  # Trick to make sure this is treated as a tensor
        self.additional_obs = (Obstacle(geometry),)

    def export_data(self, path: str, case: int, step: int, ids: tuple = None, delete_previous=True):
        """
        Export data to files

        Params:
            path: path to directory that the data will be exported to
            case: used for file nomenclature
            step: used for file nomenclature
            ids: which variables to save
            delete_previous: if True all files on folder will be deleted

        """
        export_funcs = dict(
            pressure=lambda: self.pressure.values.native().detach().cpu().numpy(),
            vx=lambda: self.velocity.x.data.native().detach().cpu().numpy(),
            vy=lambda: self.velocity.y.data.native().detach().cpu().numpy(),
            obs_xy=lambda: self.obstacle.geometry.center.native().detach().cpu().numpy(),
            obs_vx=lambda: self.obstacle.velocity.native().detach().cpu().numpy()[0],
            obs_vy=lambda: self.obstacle.velocity.native().detach().cpu().numpy()[1],
            obs_ang=lambda: self.obstacle.geometry.angle.native().view(1).detach().cpu().numpy() - PI / 2,
            obs_ang_vel=lambda: self.obstacle.angular_velocity.native().view(1).detach().cpu().numpy(),
            obs_mask=lambda: (HardGeometryMask(self.obstacle.geometry) >> self.pressure).data.native().detach().cpu().numpy(),
            fluid_force_x=lambda: self.fluid_force.native().detach().cpu().numpy()[0],
            fluid_force_y=lambda: self.fluid_force.native().detach().cpu().numpy()[1],
            fluid_torque_x=lambda: self.fluid_torque.native().detach().cpu().numpy()[0],
            fluid_torque_y=lambda: self.fluid_torque.native().detach().cpu().numpy()[1],
            fluid_torque=lambda: math.sum(self.fluid_torque).native().detach().cpu().numpy(),
            obs2_xy=lambda: self.additional_obs[0].geometry.center.native().detach().cpu().numpy(),
        )
        if not ids: ids = export_funcs.keys()
        os.makedirs(path, exist_ok=True)
        if delete_previous: shutil.rmtree(f"{path}/data/", ignore_errors=True)
        os.makedirs(f"{path}/data/", exist_ok=True)
        for var in ids:
            if var in export_funcs.keys(): data = export_funcs[var]()
            else:
                try:
                    data = getattr(self, var)
                    if torch.is_tensor(data):
                        data = data.detach().cpu().numpy()
                except:
                    print(f"Could not save variable {var}")
                    continue
            os.makedirs(f"{path}/data/{var}", exist_ok=True)
            np.save(f"{path}/data/{var}/{var}_case{case:04d}_{step:04d}.npy", data)

    def apply_forces(self, additional_force: torch.Tensor = 0, additional_torque: torch.Tensor = 0):
        """
        Update obstacle velocity after applying forces on it

        Params:
            additional_force: additional force which will be added to calculation
            additional_torque: additional torque that will be added to calculation

        """
        force = self.fluid_force + additional_force
        acc = force / self.obs_mass
        torque = math.sum(self.fluid_torque) + additional_torque
        angular_acc = torque / self.obs_inertia * (2 * math.PI)  # Radians
        # Integrate accelerations
        new_velocity = self.obstacle.velocity + acc * self.dt
        new_ang_velocity = self.obstacle.angular_velocity + angular_acc * self.dt
        self.obstacle = self.obstacle.copied_with(velocity=new_velocity, angular_velocity=new_ang_velocity.vector[0] * (not self.translation_only))

    def calculate_fluid_forces(self):
        """
        Calculate fluid forces on obstacle

        """
        # Mask gradient outside obstacle
        active = self.domain.grid(HardGeometryMask(self.obstacle.geometry), extrapolation=self.domain.boundaries["active_extrapolation"])
        accessible = self.domain.grid(active, extrapolation=math.extrapolation.ConstantExtrapolation(0))
        hard_bcs = field.stagger(accessible, math.maximum, self.domain.boundaries["scalar_extrapolation"], type=StaggeredGrid)
        # Make sure pressure inside obstacle is 0
        pressure_with_bcs = self.pressure * (1 - accessible.at(self.pressure))
        pressure_with_normal = spatial_gradient(pressure_with_bcs, StaggeredGrid,) * pressure_with_bcs.dx
        pressure_surface = pressure_with_normal * hard_bcs
        # Force
        self.fluid_force = math.channel_stack((-math.sum(pressure_surface.x.values), -math.sum(pressure_surface.y.values)), "vector")
        # Torque
        points = pressure_surface.points.unstack("staggered")
        lever = [points[0].unstack("vector")[1] - self.obstacle.geometry.center[1], points[1].unstack("vector")[0] - self.obstacle.geometry.center[0]]
        lever = math.channel_stack(lever, "vector")
        lever = self.domain.staggered_grid(lever)
        torque = self.domain.staggered_grid((1, -1)) * pressure_surface * lever
        self.fluid_torque = math.tensor((math.sum(torque.x.data), math.sum(torque.y.data)))

    def detach_variables(self):
        """
        Detach all variables from graph

        """
        self.fluid_force = math.tensor(self.fluid_force.native().detach().clone())
        self.fluid_torque = self.fluid_torque.native().detach().clone()
        self.pressure = self.domain.scalar_grid(self.pressure.values.native().detach().clone())
        vx = math.tensor(self.velocity.x.values.native().detach().clone(), ("x", "y"))
        vy = math.tensor(self.velocity.y.values.native().detach().clone(), ("x", "y"))
        self.velocity = self.domain.staggered_grid(math.channel_stack((vx, vy), "vector"))
        obs_xy = self.obstacle.geometry.center.native().detach().clone()
        obs_ang = self.obstacle.geometry.angle.native().detach().clone().view(1)
        obs_vx = math.tensor(self.obstacle.velocity.native().detach().clone())[0]
        obs_vy = math.tensor(self.obstacle.velocity.native().detach().clone())[1]
        obs_ang_vel = self.obstacle.angular_velocity.native().detach().clone().view(1)
        box_coordinates = [slice(obs_xy[0] - self.ic['obs_w'] / 2, obs_xy[0] + self.ic['obs_w'] / 2,),
                           slice(obs_xy[1] - self.ic['obs_h'] / 2, obs_xy[1] + self.ic['obs_h'] / 2,), ]
        obstacle_geometry = Box[box_coordinates].rotated(obs_ang[0])
        self.obstacle = Obstacle(
            obstacle_geometry,
            velocity=math.tensor((obs_vx, obs_vy)),
            angular_velocity=math.tensor(obs_ang_vel[0], convert=True),
        )


if __name__ == "__main__":
    # Test
    simulator = TwoWayCouplingSimulation()
    initial_i = simulator.set_initial_conditions(10, 5, obs_xy=(20, 30))
    # initial_i = simulator.set_initial_conditions(obs_width, obs_height, path=sim_load_path)
    simulator.setup_world((60, 60), 0.1, 50, 1000, 2)
    for i in range(initial_i, 500):
        simulator.advect()
        simulator.make_incompressible()
        simulator.calculate_fluid_forces()
        simulator.apply_forces()
        if i % 10 == 0:
            print(i)
            simulator.export_data("/home/ramos/test/", 0, int(i / 10), delete_previous=i == 0)
    print("Done")

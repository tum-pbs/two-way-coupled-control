from phi.torch.flow import *
import sys
import gc
import os
from natsort import natsorted
import shutil
import copy
from phi.field import spatial_gradient
TORCH_BACKEND.set_default_device("GPU")


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

    def __init__(self, translation_only: bool = False):
        """
        Class initializer. If translation_only is True, simulation won't have rotation

        """
        self.ic = {}
        self.translation_only = translation_only
        self.additional_obs = []

    def set_initial_conditions(
        self,
        obs_w: float,
        obs_h: float,
        path: str = None,
        obs_xy: math.Tensor = None,
        obs_ang: math.Tensor = math.tensor((PI / 2,), convert=True),
        obs_vel: math.Tensor = math.tensor([0, 0]),
        obs_ang_vel: math.Tensor = math.tensor((0.0,), convert=True),
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
            for variable in [
                "pressure",
                "vx",
                "vy",
                "obs_xy",
                "obs_vx",
                "obs_vy",
                "obs_ang",
                "obs_ang_vel",
            ]:
                self.ic[variable] = np.load(
                    f"{path}/data/{variable}/{variable}_{case_snapshot}")
                if variable in ["vx", "vy"]:
                    self.ic[variable] = math.tensor(
                        self.ic[variable], ("x", "y"))
            i0 = int(case_snapshot.split('_')[0][4:])
            return i0
        else:
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
        self.fluid_force = math.tensor(torch.zeros(2).cuda())
        self.fluid_torque = math.tensor(torch.zeros(1).cuda())
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
        self.additional_obs = (Obstacle(Sphere(xy, radius)),)

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
        if not ids:
            ids = (
                "pressure",
                "vx",
                "vy",
                "obs_mask",
                "obs_xy",
                "obs_vx",
                "obs_vy",
                "obs_ang",
                "obs_ang_vel",
                "fluid_force_x",
                "fluid_force_y",
                "fluid_torque_x",
                "fluid_torque_y",
            )
        if not os.path.exists((path)):
            os.mkdir(path)
        if delete_previous:
            shutil.rmtree(f"{path}/data/", ignore_errors=True)
        if not os.path.exists(f"{path}/data/"):
            os.mkdir(f"{path}/data/")
        for var in ids:
            if not os.path.exists(f"{path}/data/{var}"):
                os.mkdir(f"{path}/data/{var}")
            if var == "pressure":
                data = self.pressure.values.native().detach().cpu().numpy()
            elif var == "vx":
                data = self.velocity.x.data.native().detach().cpu().numpy()
            elif var == "vy":
                data = self.velocity.y.data.native().detach().cpu().numpy()
            elif var == "obs_xy":
                data = self.obstacle.geometry.center.native().detach().cpu().numpy()
            elif var == "obs_vx":
                data = self.obstacle.velocity.native().detach().cpu().numpy()[0]
            elif var == "obs_vy":
                data = self.obstacle.velocity.native().detach().cpu().numpy()[1]
            elif var == "obs_ang":
                data = self.obstacle.geometry.angle.native().view(1).detach().cpu().numpy()
            elif var == "obs_ang_vel":
                data = self.obstacle.angular_velocity.native().view(1).detach().cpu().numpy()
            elif var == "obs_mask":
                data = (HardGeometryMask(self.obstacle.geometry) >>
                        self.pressure).data.native().detach().cpu().numpy()
            elif var == 'fluid_force_x':
                data = self.fluid_force.native().detach().cpu().numpy()[0]
            elif var == 'fluid_force_y':
                data = self.fluid_force.native().detach().cpu().numpy()[1]
            elif var == 'fluid_torque_x':
                data = self.fluid_torque.native().detach().cpu().numpy()[0]
            elif var == 'fluid_torque_y':
                data = self.fluid_torque.native().detach().cpu().numpy()[1]
            elif var == 'fluid_torque':
                data = math.sum(self.fluid_torque).native().detach().cpu().numpy()
            elif var == "max_cfl":
                data = (math.max(math.abs((self.velocity >> self.pressure).values)) * self.dt).native().detach().cpu().numpy()
            else:
                try:
                    data = getattr(self, var)
                    if torch.is_tensor(data):
                        data = data.detach().cpu().numpy()
                except:
                    print(f"Could not save variable {var}")
                    continue
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

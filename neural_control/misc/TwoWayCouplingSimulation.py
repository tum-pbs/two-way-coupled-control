from collections import defaultdict
from phi.math.extrapolation import ConstantExtrapolation, Extrapolation, _BoundaryExtrapolation

from torch._C import device
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
        obs_vel: list = [0., 0.],
        obs_ang_vel: float = 0.,
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
        self.ic["obs_w"] = torch.as_tensor(obs_w).to(self.device)
        self.ic["obs_h"] = torch.as_tensor(obs_h).to(self.device)
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
                self.ic[var] = torch.as_tensor(np.load(f"{path}/data/{var}/{var}_{case_snapshot}")).to(self.device)
                if var in ["vx", "vy"]: self.ic[var] = math.tensor(self.ic[var], ("x", "y"))
                if var == 'obs_ang': self.ic[var] += PI / 2  # Account for difference in angle convention
            # Try to load a second obstacle
            try:
                self.ic['obs2_xy'] = torch.as_tensor(np.load(f"{path}/data/obs2_xy/obs2_xy_case0000_0000.npy"))
            except:
                print("Did not found data of second obstacle")
                pass
            i0 = int(case_snapshot.split('_')[1][:4])
            return i0
        else:
            self.ic["pressure"] = 0
            self.ic["vx"] = self.ic["vy"] = math.tensor(0.)
            # self.ic["obs_xy"] = obs_xy
            self.ic["obs_xy"] = torch.as_tensor(obs_xy)
            self.ic["obs_vx"] = obs_vel[0]
            self.ic["obs_vy"] = obs_vel[1]
            self.ic["obs_ang_vel"] = math.tensor((torch.as_tensor(obs_ang_vel),), convert=True)
            self.ic["obs_ang"] = math.tensor((torch.as_tensor(obs_ang),), convert=True)
            return 0

    def setup_world(self, re: float, domain_size: list, dt: float, obs_mass: float, obs_inertia: float, inflow_velocity: float, sponge_intensity: float = 0.0, sponge_size: list = [0, 0, 0, 0]):
        """
        Setup world for simulation with a box on it

        Params:
            re : Reynolds number
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
        open_bc = {
            "accessible_extrapolation": extrapolation.ConstantExtrapolation(1),
            "active_extrapolation": extrapolation.ConstantExtrapolation(0),
            "near_vector_extrapolation": extrapolation.ZERO,
            "scalar_extrapolation": extrapolation.ZERO,
            "vector_extrapolation": extrapolation.BOUNDARY
        }
        self.domain = Domain(
            x=domain_size[0],
            y=domain_size[1],
            boundaries=((open_bc, open_bc), (open_bc, open_bc)),
            bounds=Box[0: domain_size[0], 0: domain_size[1]],
        )
        self.pressure = self.domain.scalar_grid(self.ic["pressure"])
        self.velocity = self.domain.staggered_grid(
            math.channel_stack((self.ic["vx"], self.ic["vy"]), "vector"))
        # Obstacle
        lower = torch.as_tensor([self.ic["obs_xy"][0] - self.ic["obs_w"] / 2, self.ic["obs_xy"][1] - self.ic["obs_h"] / 2]).to(self.device)
        upper = torch.as_tensor([self.ic["obs_xy"][0] + self.ic["obs_w"] / 2, self.ic["obs_xy"][1] + self.ic["obs_h"] / 2]).to(self.device)
        obstacle_geometry = Box(lower, upper).rotated(self.ic["obs_ang"][0])
        self.obstacle = Obstacle(
            obstacle_geometry,
            velocity=math.tensor(torch.as_tensor((self.ic["obs_vx"], self.ic["obs_vy"])).to(self.device)),
            angular_velocity=math.tensor(self.ic["obs_ang_vel"][0], convert=True),
        )
        # Add additional obstacle if it was provided
        try:
            self.add_box(self.ic["obs2_xy"], self.ic["obs_w"], self.ic["obs_w"])
        except:
            print("\n Only one obstacle in simulation")
            pass
        # Constant velocity
        self.inflow_velocity_mask = HardGeometryMask(Box[:0.5, :]) >> self.velocity
        self.velocity += self.inflow_velocity_mask
        self.velocity -= self.inflow_velocity_mask
        self.inflow_velocity = inflow_velocity
        self.re = re
        self.viscosity = inflow_velocity * self.ic["obs_w"] / re
        # Sponge masks
        points_uv = self.velocity.points.unstack("staggered")
        # Sponge on the right boundary has a ramp in the u-direction
        placeholder = [[], [], []]
        # Create a linear ramp in the outward direction
        for points in points_uv:
            # Bottom boundary
            points_y = -(points.vector[1] - sponge_size[1])
            sponge = (math.abs(points_y) + points_y) / 2  # Make negatives be 0
            sponge = sponge / sponge_size[1]
            placeholder[0] += [sponge]
            # Right boundary
            points_x = points.vector[0] - (domain_size[0] - sponge_size[2])
            sponge = (math.abs(points_x) + points_x) / 2  # Make negatives be 0
            sponge = sponge / sponge_size[2]  # Normalize
            placeholder[1] += [sponge]
            # Top boundary
            points_y = points.vector[1] - (domain_size[1] - sponge_size[3])
            sponge = (math.abs(points_y) + points_y) / 2  # Make negatives be 0
            sponge = sponge / sponge_size[3]
            placeholder[2] += [sponge]
        masks = [self.domain.staggered_grid(math.channel_stack(values, "vector")) for values in placeholder]
        self.sponge_normal_mask = (masks[0] > 0) * (0, -1) + (masks[1] * (1, 0) > 0) + (masks[2] * (0, 1) > 0)
        self.sponge_mask = (masks[0] > 0) * (0, 1) + (masks[1] * (1, 0) > 0) + (masks[2] * (0, 1) > 0)
        self.sponge = (masks[0] * (0, 1) + masks[1] * (1, 0) + masks[2] * (0, 1)) * sponge_intensity * dt
        # self.sponge = (self.sponge_normal_mask * self.sponge_normal_mask) * sponge_intensity * dt

    def advect(self, tripping_on: bool = False):
        """
        Perform advection step of fluid and obstacle

        Params:
            tripping_on: if True, then inflow is multiplied by mask with random values between 0.5 and 1

        """
        def rhs(velocity):
            u = velocity.x
            v = velocity.y
            du_dx, du_dy = spatial_gradient(u, CenteredGrid).unstack("vector")
            dv_dx, dv_dy = spatial_gradient(v, CenteredGrid).unstack("vector")
            v_at_u = (v.values[:, 1:] + v.values[:, :-1]) / 2.  # v at center node
            v_at_u = (v_at_u[1:, :] + v_at_u[:-1, :]) / 2.  # Values at u nodes without boundaries
            bc_v_left = math.zeros(x=1, y=v_at_u.shape[1])  # BC for v at left boundary
            bc_v_right = v_at_u[-2:-1, :]  # Neumman BC for v at right boundary
            v_at_u = math.concat((bc_v_left, v_at_u, bc_v_right), 'x')  # v at u nodes
            u_at_v = (u.values[1:, :] + u.values[:-1, :]) / 2.  # u at center nodes
            u_at_v = (u_at_v[:, 1:] + u_at_v[:, :-1]) / 2.
            bc_u_top = u_at_v[:, -2:-1]
            bc_u_bottom = u_at_v[:, 0:1]
            u_at_v = math.concat((bc_u_bottom, u_at_v, bc_u_top), 'y')
            result = math.channel_stack([
                -1. * (u.values * du_dx.values + v_at_u * du_dy.values),
                -1. * (u_at_v * dv_dx.values + v.values * dv_dy.values)], "vector")
            result = self.domain.staggered_grid(result, extrapolation=velocity.extrapolation)
            return result  # , du_dx.values.native().cpu().numpy(), du_dy.values.native().cpu().numpy(), dv_dx.values.native().cpu().numpy(), dv_dy.values.native().cpu().numpy()

        # 4th order RK
        # rhs1, *_ = rhs(self.velocity)
        # u1 = self.velocity + 0.5 * self.dt * rhs1
        # rhs2, *_ = rhs(u1)
        # u2 = self.velocity + 0.5 * self.dt * rhs2
        # rhs3, *_ = rhs(u2)
        # u3 = self.velocity + self.dt * rhs3
        # rhs4, self.du_dx, self.du_dy, self.dv_dx, self.dv_dy = rhs(u3)
        # convection_term = self.velocity + 1 / 6. * self.dt * (rhs1 + 2 * (rhs2 + rhs3) * rhs4)
        # 2nd order RK
        rhs1 = rhs(self.velocity)
        k1 = self.dt * rhs1
        rhs2 = rhs(self.velocity + k1)
        k2 = self.dt * rhs2
        convection_term = self.velocity + (k1 + k2) / 2.
        # Diffusive term
        velocity_laplace = math.laplace(self.velocity.values)
        diffusion_term = velocity_laplace * self.viscosity
        # self.velocity = advect.semi_lagrangian(self.velocity, self.velocity, self.dt) + diffusion_term
        self.velocity = self.velocity.copied_with(values=convection_term.values + diffusion_term, extrapolation=self.velocity.extrapolation)
        # Apply sponge
        # If the velocity is aligned with boundary outward normal, deaccelerate, otherwise set it to 0
        vel_boundaries = self.sponge_normal_mask * self.velocity  # Project velocity onto boundaries normal directions at sponge
        vel_boundaries = vel_boundaries - self.sponge  # Apply sponge deacceleration
        vel_boundaries = vel_boundaries * (vel_boundaries > 0)   # Avoid creating flow in opposite normal direction
        vel_boundaries = vel_boundaries * self.sponge_normal_mask  # Project velocity back
        self.velocity = self.velocity * (1 - self.sponge_mask) + vel_boundaries
        # Tripping
        if tripping_on:
            tripping_x = math.random_uniform(self.velocity.x.shape) * 0.5 + 0.5
            tripping_y = math.zeros(self.velocity.y.shape)
            tripping = math.channel_stack((tripping_x, tripping_y), "vector")
        else: tripping = 1
        self.velocity = self.velocity * (1 - self.inflow_velocity_mask.values) + self.inflow_velocity_mask.values * (self.inflow_velocity, 0) * tripping
        # Obstacle advection
        new_geometry = self.obstacle.geometry.rotated(-self.obstacle.angular_velocity * self.dt)
        new_geometry = new_geometry.shifted(self.obstacle.velocity * self.dt)
        self.obstacle = self.obstacle.copied_with(geometry=new_geometry, age=self.obstacle.age + self.dt)

    def make_incompressible(self):
        """
        Ensure divergence free condition on velocity field

        """
        new_velocity, new_pressure, *_ = fluid.make_incompressible(
            self.velocity,
            self.domain,
            (self.obstacle, *self.additional_obs),
            solve_params=self.solve_params,
            pressure_guess=self.pressure,
        )
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
        lower = torch.as_tensor([xy[0] - width / 2, xy[1] - height / 2]).to(self.device)
        upper = torch.as_tensor([xy[0] + width / 2, xy[1] + height / 2]).to(self.device)
        geometry = Box(lower, upper).shifted(torch.tensor(0).to(self.device))  # Trick to make sure this is treated as a tensor
        self.additional_obs += [Obstacle(geometry), ]

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
        def calculate_vorticity():
            dvy_shifted = math.gradient(self.velocity.y.values, dx=1., difference='central', dims='x').spatial_gradient[0]
            dvx_shifted = math.gradient(self.velocity.x.values, dx=1., difference='central', dims='y').spatial_gradient[0]
            dvy_center = (dvy_shifted[:, 1:] + dvy_shifted[:, :-1]) / 2
            dvx_center = (dvx_shifted[1:, :] + dvx_shifted[:-1, :]) / 2
            return dvy_center - dvx_center

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
            obs2_xy=lambda: [obs.geometry.center.native().detach().cpu().numpy() for obs in self.additional_obs],
            cfl=lambda: (math.max(math.abs(self.velocity.values)) * self.dt).native().detach().cpu().numpy(),
            vorticity=lambda: calculate_vorticity().native().detach().cpu().numpy(),
        )
        if not ids: ids = list(export_funcs.keys())
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
            np.save(os.path.abspath(f"{path}/data/{var}/{var}_case{case:04d}_{step:04d}.npy"), data)

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
        lower = torch.as_tensor([obs_xy[0] - self.ic["obs_w"] / 2, obs_xy[1] - self.ic["obs_h"] / 2]).to(self.device)
        upper = torch.as_tensor([obs_xy[0] + self.ic["obs_w"] / 2, obs_xy[1] + self.ic["obs_h"] / 2]).to(self.device)
        obstacle_geometry = Box(lower, upper).rotated(obs_ang[0])
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

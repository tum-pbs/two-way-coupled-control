from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
import shutil
from misc.TwoWayCouplingSimulation import TwoWayCouplingSimulation
from misc_funcs import extract_inputs, Probes, prepare_export_folder, rotate
from reinforcement_learning.envs.numpy_wrapper import NumpyWrapper
from reinforcement_learning.envs.seed_on_reset_wrapper import SeedOnResetWrapper
from reinforcement_learning.envs.skip_stack_wrapper import BrenerStacker
from gym import Env
from gym.spaces import Box
from phi import math
import os
from InputsManager import InputsManager
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CallbackList
from reinforcement_learning.callbacks import EveryNRolloutsPlusStartFinishFunctionCallback


class TwoWayCouplingTorchEnv(Env):
    def __init__(
        self, 
        device: str,
        n_steps: int,
        dt: float,
        domain_size: Tuple[int, int],
        destination_margins: Tuple[int, int],
        re: float,
        obs_type: str,
        obs_width: float, 
        obs_height: float, 
        obs_xy: Tuple[int, int], 
        obs_mass: float,
        obs_inertia: float,
        translation_only: bool,
        sponge_intensity: float,
        sponge_size: List[int],
        inflow_on: bool,
        inflow_velocity: float,
        probes_offset: float, 
        probes_size: float,
        probes_n_rows: int,
        probes_n_columns: int,
        sim_import_path: str,
        sim_export_path: str,
        export_vars: List[str],
        export_stride: int,
        input_vars: list,
        ref_vars: dict,
    ):
        self.sim = TwoWayCouplingSimulation(device, translation_only)
        print(f"Sim import path: {sim_import_path}")
        self.sim.set_initial_conditions(
            obs_type=obs_type,
            obs_w=obs_width, 
            obs_h=obs_height, 
            path=sim_import_path
        )
        self.dt = dt
        self.destination_margins = destination_margins
        self.domain_size = domain_size
        self.re = re
        self.obs_mass = obs_mass
        self.obs_inertia = obs_inertia
        self.translation_only = translation_only
        self.sponge_intensity = sponge_intensity
        self.sponge_size = sponge_size
        self.input_vars = input_vars
        self.ref_vars = ref_vars
        self.inflow_on = inflow_on
        self.inflow_velocity = inflow_velocity

        self.probes = Probes(
            width_inner=obs_width / 2 + probes_offset,
            height_inner=obs_height / 2 + probes_offset,
            size=probes_size,
            n_rows=probes_n_rows,
            n_columns=probes_n_columns,
            center=obs_xy,
        )

        self.n_steps = n_steps
        self.step_idx = 0
        self.epis_idx = -1

        self.pos_objective = None
        self.ang_objective = None
        self.forces = None
        self.torque = None
        self.pos_error = None
        self.ang_error = None
        self.rew = None
        self.rew_baseline = torch.tensor(0)

        self.sim_export_path = sim_export_path
        self.export_vars = export_vars
        self.export_stride = export_stride
        self.export_folder_created = False

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    @property
    def _obstacle_angle(self) -> torch.Tensor:
        return self.sim.obstacle.geometry.angle.native() - math.PI / 2.0

    def reset(self)-> torch.Tensor:
        self.step_idx = 0
        self.epis_idx += 1
        self.sim.setup_world(
            re=self.re, 
            domain_size=self.domain_size, 
            dt=self.dt, 
            obs_mass=self.obs_mass, 
            obs_inertia=self.obs_inertia, 
            reference_velocity=self.inflow_velocity, 
            sponge_intensity=self.sponge_intensity,
            sponge_size=self.sponge_size,
            inflow_on=self.inflow_on,
        )
        self.pos_objective, self.ang_objective = self._generate_objectives()
        obs, loss = self._extract_inputs()
        self.rew_baseline = self._get_rew(loss, False)
        return obs

    def step(self, action: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, bool, Dict[str, Any]]:
        self.step_idx += 1
        self.forces, self.torque = self._split_action_to_force_torque(action.to(self.sim.device))
        self.forces = self._to_global(self.forces)

        self.sim.apply_forces(self.forces * self.ref_vars['force'], self.torque * self.ref_vars['torque'])
        self.sim.advect()
        self.sim.make_incompressible()
        self.probes.update_transform(self.sim.obstacle.geometry.center.native(), -1 * self._obstacle_angle)
        self.sim.calculate_fluid_forces()
        obs, loss = self._extract_inputs()
        done = self._obstacle_leaving_domain() or self.step_idx == self.n_steps

        if torch.isnan(torch.sum(obs)):
            print('NaN value in observation!')
            obs[torch.isnan(obs)] = 0
            done = True

        if torch.sum(self.sim.obstacle.velocity.native() ** 2) > self.ref_vars['max_vel'] ** 2:
            print('Hit maximum velocity, ending trajectory')
            done = True

        if not self.translation_only and torch.abs(self.sim.obstacle.angular_velocity.native()) > self.ref_vars['max_ang_vel']:
            print('Hit maximum angular velocity, ending trajectory')
            done = True

        self.rew = self._get_rew(loss, done, self.rew_baseline)
        info = {}

        return obs, self.rew, done, info

    def render(self):
        if not self.export_folder_created:
            self.epis_idx = 0       # Reset episode index for interactive data reader to work properly
            shutil.rmtree(f"{self.sim_export_path}/tensorboard", ignore_errors=True)
            prepare_export_folder(self.sim_export_path, self.step_idx)
            self.export_folder_created = True
        
        probes_points = self.probes.get_points_as_tensor()
        self.sim.probes_points = probes_points.native().detach()
        self.sim.probes_vx = self.sim.velocity.x.sample_at(probes_points).native().detach()
        self.sim.probes_vy = self.sim.velocity.y.sample_at(probes_points).native().detach()
        self.sim.control_force_x, self.sim.control_force_y = self.forces.detach().clone() * self.ref_vars['force']
        self.sim.control_torque = self.torque.detach().clone() * self.ref_vars['torque']
        self.sim.reference_x = self.pos_objective[0].detach().clone()
        self.sim.reference_y = self.pos_objective[1].detach().clone()
        self.sim.reference_angle = self.ang_objective.detach().clone()
        self.sim.error_x = self.pos_error[0]
        self.sim.error_y = self.pos_error[1]
        if not self.translation_only:
            self.sim.error_ang = self.ang_error
        self.sim.reward = self.rew
        
        self.sim.export_data(
            self.sim_export_path, 
            self.epis_idx, 
            self.step_idx // self.export_stride, 
            self.export_vars, 
            (self.epis_idx==0 and self.step_idx == 0)
        )

    def seed(self, seed=0) -> None:
        print("this is seed, yo")
        torch.manual_seed(seed)

    def _get_action_space(self) -> Box:
        dim = 2
        if not self.translation_only:
            dim += 1
        return Box(-1, 1, shape=(dim,), dtype=np.float32)

    def _get_observation_space(self) -> Box:
        shape = self.reset().shape
        self.epis_idx -= 1  # Account for reset function call
        return Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    def _generate_objectives(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_objective_min = torch.tensor(self.destination_margins)
        pos_objective_max = torch.tensor(self.domain_size) - torch.tensor(self.destination_margins)
        pos_objective = pos_objective_min + (torch.rand(2) * (pos_objective_max - pos_objective_min))
        ang_objective = torch.rand(1) * 2 * math.PI - math.PI
        return pos_objective.to(self.sim.device), ang_objective.to(self.sim.device)

    def _extract_inputs(self) -> Tuple[torch.Tensor, dict]:
        obs, loss = extract_inputs(self.input_vars, self.sim, self.probes, self.pos_objective, self.ang_objective, self.ref_vars, self.translation_only)
        return obs.reshape(-1), loss

    def _get_rew(self, loss_inputs: dict, done: bool, baseline: Optional[torch.Tensor]=None) -> torch.Tensor:
        self.pos_error = torch.stack([loss_inputs[key] for key in ['error_x', 'error_y']])

        pos_rew = -1 * torch.sum(self.pos_error ** 2)

        ang_vel_rew = -10 * (self.sim.obstacle.angular_velocity.native() / self.ref_vars['max_ang_vel']) ** 2

        rew = pos_rew + ang_vel_rew
        if baseline:
            rew = (rew - baseline) / torch.abs(baseline)

        if torch.sum(self.pos_error ** 2) < 0.15 ** 2:
            rew += 9

        rew = torch.clamp(rew, min=-30)

        if not self.translation_only:
            pass
            # TODO calculate angular reward and add to output

        return rew

    def _obstacle_leaving_domain(self) -> bool:
        obstacle_center = self.sim.obstacle.geometry.center
        return math.any(obstacle_center > self.domain_size) or math.any(obstacle_center < (0, 0))


    def _split_action_to_force_torque(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        control_effort = action.to(self.sim.device)
        force = control_effort[:2]
        torque = torch.tensor([0]).to(self.sim.device) if self.translation_only else control_effort[-1:]
        return force, torque

    def _to_global(self, force: torch.Tensor) -> torch.Tensor:
        return rotate(force, -1 * self._obstacle_angle)


class TwoWayCouplingConfigTorchEnv(TwoWayCouplingTorchEnv):
    def __init__(self, config_path):
        simulation_storage_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'storage', 'simulation_data')
        config = InputsManager(config_path)
        config.calculate_properties()

        device = config.device
        max_acc = config.max_acc
        max_vel = config.max_vel
        max_ang_acc = config.max_ang_acc
        max_ang_vel = config.max_ang_vel
        translation_only = config.translation_only
        probes_offset = config.probes_offset
        probes_size = config.probes_size
        probes_n_rows = config.probes_n_rows
        probes_n_columns = config.probes_n_columns
        export_vars = config.export_vars
        export_stride = config.export_stride

        n_steps = config.online['n_timesteps']
        destination_margins = config.online['destinations_margins']
        sim_import_path = os.path.join(simulation_storage_path, config.online['simulation_path'])
        sim_export_path = os.path.join(simulation_storage_path, config.online['export_path'])

        dt = config.simulation['dt']
        domain_size = config.simulation['domain_size']
        re = config.simulation['re']
        obs_type = config.simulation['obs_type']
        obs_width = config.simulation['obs_width']
        obs_height = config.simulation['obs_height']
        obs_xy = config.simulation['obs_xy']
        obs_mass = config.simulation['obs_mass']
        obs_inertia = config.simulation['obs_inertia']
        sponge_intensity = config.simulation['sponge_intensity']
        sponge_size = config.simulation['sponge_size']
        inflow_on = config.simulation['inflow_on']
        inflow_velocity = config.simulation['reference_velocity']

        input_vars = config.nn_vars

        ref_vars = dict(
            length=obs_width,
            angle=math.PI,
            velocity=inflow_velocity,
            ang_velocity=inflow_velocity / obs_width,
            force=obs_mass * max_acc,
            torque=obs_inertia * max_ang_acc,
            time=obs_width / inflow_velocity,
            destination_zone_size=domain_size - destination_margins * 2,
            max_vel=1 / (dt * 0.9),
            max_ang_vel=max_ang_vel,
        )

        print("Ref vars: %s" % ref_vars)

        super().__init__(
            device=device,
            n_steps=n_steps,
            dt=dt,
            domain_size=domain_size,
            destination_margins=destination_margins,
            re=re,
            obs_type=obs_type,
            obs_width=obs_width,
            obs_height=obs_height,
            obs_xy=obs_xy,
            obs_mass=obs_mass,
            obs_inertia=obs_inertia,
            translation_only=translation_only,
            sponge_intensity=sponge_intensity,
            sponge_size=sponge_size,
            inflow_on=inflow_on,
            inflow_velocity=inflow_velocity,
            probes_offset=probes_offset,
            probes_size=probes_size,
            probes_n_rows=probes_n_rows,
            probes_n_columns=probes_n_columns,
            sim_import_path=sim_import_path,
            sim_export_path=sim_export_path,
            export_vars=export_vars,
            export_stride=export_stride,
            input_vars=input_vars,
            ref_vars=ref_vars,
        )


def get_env(skip: int=8, stack: int=4) -> Env:
    inputs_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'inputs.json')
    
    env = TwoWayCouplingConfigTorchEnv(inputs_path)
    env = NumpyWrapper(env)
    env = BrenerStacker(env, 4, 4, 2, True)
    #env = SkipStackWrapper(env, skip=skip, stack=stack)
    #env = RewNormWrapper(env, None)
    env = SeedOnResetWrapper(env, 0)
    #env = Monitor(env, info_keywords=('rew_unnormalized',))
    
    print('Observation space shape: %s' % str(env.observation_space.shape))
    return env
    
def train_model(name: str, log_dir: str, n_timesteps: int, **agent_kwargs) -> SAC:
    storage_folder_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'storage')

    model_path = os.path.join(storage_folder_path, "networks", name)
    tb_log_path = os.path.join(storage_folder_path, "tensorboard", log_dir)

    env = get_env()

    print(model_path)
    if os.path.exists(model_path + '.zip'):
        print('model path exists, loading model')
        model = SAC.load(model_path, env)
    else:
        print('creating new model...')
        model = SAC('MlpPolicy', env, tensorboard_log=tb_log_path, verbose=1, **agent_kwargs)

    def store_fn(_):
        print(f"Storing model to {model_path}...")
        model.save(model_path)
        print("Stored model.")

    callback = CallbackList([
        EveryNRolloutsPlusStartFinishFunctionCallback(20000, store_fn),
    ])

    model.learn(total_timesteps=n_timesteps, callback=callback, tb_log_name=name)

if __name__ == '__main__':
    #train_model('128_128_128_3e-4_2grst_bs128_angvelpen_rewnorm_test', 'hparams_tuning', 20000, batch_size=128, learning_starts=32, learning_rate=3e-4, gradient_steps=2, policy_kwargs=dict(net_arch=[128, 128, 128]))
    train_model('brener_setup_speed_limit_long_eps', 'simple_env', 300000, batch_size=256, learning_starts=128, policy_kwargs=dict(net_arch=[38, 38]))

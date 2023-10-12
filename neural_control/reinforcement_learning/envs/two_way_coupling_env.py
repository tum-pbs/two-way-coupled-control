import shutil
from typing import Any, Dict, List, Optional, Tuple
import os

import torch
import numpy as np
from gym import Env
from gym.spaces import Box
from phi import math

from neural_control.InputsManager import InputsManager
from neural_control.misc.TwoWayCouplingSimulation import TwoWayCouplingSimulation
from neural_control.misc.misc_funcs import calculate_loss, extract_inputs, Probes, prepare_export_folder, rotate


class TwoWayCouplingEnv(Env):
    def __init__(
        self, 
        device: str,
        n_steps: int,
        dt: float,
        use_g_reward: bool,
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
        hyperparams: dict,
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
        self.use_g_reward = use_g_reward
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
        self.hyperparams = hyperparams
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
        self.rew_baseline = None

        self.sim_export_path = sim_export_path
        self.export_vars = export_vars
        self.export_stride = export_stride
        self.export_folder_created = False

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        if self.use_g_reward:
            print('\033[33mUsing G reward function\033[0m')
        else:
            print('\033[33mUsing LOVE reward function\033[0m')

    def reset(self) -> np.ndarray:
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
        obs, loss_inputs = self._extract_inputs()
        self.pos_error = np.array([val.cpu().numpy() for val in [loss_inputs[key] for key in ['error_x', 'error_y']]])
        self.rew_baseline = self._get_rew(loss_inputs, False)
        print("pos objective: %s" % str(self.pos_objective))

        assert isinstance(obs, np.ndarray)
        return obs
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        self.step_idx += 1
        self.forces, self.torque = self._split_action_to_force_torque(action)
        self.forces = self._to_global(self.forces)

        self.sim.apply_forces(self.forces * self.ref_vars['force'], self.torque * self.ref_vars['torque'])
        self.sim.advect()
        self.sim.make_incompressible()
        self.probes.update_transform(self.sim.obstacle.geometry.center.numpy(), -1 * self._obstacle_angle.numpy())
        self.sim.calculate_fluid_forces()
        obs, loss_inputs = self._extract_inputs()
        done = self._obstacle_leaving_domain() or self.step_idx == self.n_steps

        if np.isnan(np.sum(obs)):
            print('NaN value in observation!')
            obs[np.isnan(obs)] = 0
            done = True

        if math.max(math.abs(self.sim.velocity.values)) * self.dt > 1.5:
            print('Hit maximum velocity, ending trajectory')
            done = True

        self.rew = self._get_rew(loss_inputs, done, self.rew_baseline)

        info = {}

        assert isinstance(obs, np.ndarray)
        assert isinstance(self.rew, np.ndarray)
        assert isinstance(done, bool)

        return obs, self.rew, done, info

    def render(self, mode: str) -> None:
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

    def close(self) -> None:
        pass

    def seed(self, seed=0) -> None:
        torch.manual_seed(seed)

    def _generate_objectives(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_objective_min = torch.tensor(self.destination_margins)
        pos_objective_max = torch.tensor(self.domain_size) - torch.tensor(self.destination_margins)
        pos_objective = pos_objective_min + (torch.rand(2) * (pos_objective_max - pos_objective_min))
        ang_objective = torch.rand(1) * 2 * math.PI - math.PI
        return pos_objective.to(self.sim.device), ang_objective.to(self.sim.device)

    def _get_action_space(self) -> Box:
        dim = 2
        if not self.translation_only:
            dim += 1
        return Box(-1, 1, shape=(dim,), dtype=np.float32)

    def _get_observation_space(self) -> Box:
        shape = self.reset().shape
        self.epis_idx -= 1  # Account for reset function call
        return Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    def _extract_inputs(self) -> Tuple[np.ndarray, dict]:
        current_input_vars = [var for var in self.input_vars if 'control' not in var]
        obs, loss_inputs = extract_inputs(self.input_vars, self.sim, self.probes, self.pos_objective, self.ang_objective, self.ref_vars, self.translation_only)

        if self.forces is not None:
            forces = self.forces.detach().clone()
        else:
            forces = torch.zeros(4, device=self.sim.device)

        loss_inputs['control_force_x'] = forces[0:1]
        loss_inputs['control_force_y'] = forces[1:2]
        for dim in ['x', 'y']:
            loss_inputs[f'd_control_force_{dim}'] = loss_inputs[f"control_force_{dim}"]
            
        return obs.cpu().numpy().reshape(-1), loss_inputs

    def _get_obs(self) -> np.ndarray:
        return self._extract_inputs()[0]

    def _g_rew(self, loss_inputs: dict, done: bool, baseline: Optional[np.ndarray]=None) -> np.ndarray:
        self.pos_error = np.array([val.cpu().numpy() for val in [loss_inputs[key] for key in ['error_x', 'error_y']]])
        rew = -1 * np.sum(self.pos_error ** 2)

        if baseline:
            rew = (rew - baseline) / np.abs(baseline)

        if np.sum(self.pos_error ** 2) < 0.15 ** 2:
            rew += 9

        #rew = np.max([rew, -30])
        return np.array(rew)

    def _love_rew(self, loss_inputs: dict, done: bool, baseline: Optional[np.ndarray]=None) -> np.ndarray:
        loss, _ = calculate_loss(loss_inputs, self.hyperparams, self.translation_only)
        rew = -1 * loss

        if done and self.step_idx != self.n_steps:
            rew -= 1000
        
        return rew.cpu().numpy()

    def _get_rew(self, loss_inputs: dict, done: bool, baseline: Optional[np.ndarray]=None) -> np.ndarray:
        if self.use_g_reward:
            return self._g_rew(loss_inputs, done, baseline)
        else:
            return self._love_rew(loss_inputs, done, baseline)

    def _obstacle_leaving_domain(self) -> bool:
        obstacle_center = self.sim.obstacle.geometry.center
        return bool(math.any(obstacle_center > self.domain_size) or math.any(obstacle_center < (0, 0)))

    def _split_action_to_force_torque(self, action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        control_effort = torch.tensor(action).to(self.sim.device)
        torque = torch.tensor([0]).to(self.sim.device) if self.translation_only else control_effort[-1:]
        return control_effort[:2], torque

    def _to_global(self, force: torch.Tensor) -> torch.Tensor:
        return rotate(force, -1 * self._obstacle_angle)

    @property
    def _obstacle_angle(self) -> torch.Tensor:
        return (self.sim.obstacle.geometry.angle - math.PI / 2.0).native().cpu()


class TwoWayCouplingConfigEnv(TwoWayCouplingEnv):
    def __init__(self, config_path):
        config = InputsManager(config_path)
        config.calculate_properties()

        device = config.device
        max_acc = config.max_acc
        max_ang_acc = config.max_ang_acc
        translation_only = config.translation_only
        probes_offset = config.probes_offset
        probes_size = config.probes_size
        probes_n_rows = config.probes_n_rows
        probes_n_columns = config.probes_n_columns
        export_vars = config.export_vars
        export_stride = config.export_stride

        n_steps = config.rl['n_timesteps']
        use_g_reward = config.rl['use_g_reward']
        destination_margins = np.array(config.rl['destinations_margins'])
        hyperparams = config.rl['loss_hyperparams']
        sim_import_path = config.rl['simulation_path']
        sim_export_path = config.rl['export_path']

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
            velocity=inflow_velocity,
            length=obs_width,
            force=obs_mass * max_acc,
            angle=math.PI,
            torque=obs_inertia * max_ang_acc,
            time=obs_width / inflow_velocity,
            ang_velocity=inflow_velocity / obs_width,
        )

        print("Ref vars: %s" % ref_vars)

        super().__init__(
            device=device,
            n_steps=n_steps,
            dt=dt,
            use_g_reward=use_g_reward,
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
            hyperparams=hyperparams,
        )

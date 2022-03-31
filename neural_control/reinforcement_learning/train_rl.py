import os
import shutil
from argparse import ArgumentParser

from gym import Env
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from phi import math

from neural_control.InputsManager import InputsManager

from envs.two_way_coupling_env import TwoWayCouplingConfigEnv
from envs.stack_observations_wrapper import StackObservations
from envs.seed_on_reset_wrapper import SeedOnResetWrapper
from extract_model import store_sac_actor_as_torch_module
from callbacks import EveryNTimestepsPlusStartFinishFunctionCallback


CONFIG_FILENAME = 'inputs.json'
AGENT_FILENAME = 'agent.zip'
TORCH_MODEL_FILENAME_TEMPLATE = 'trained_model_%04i.pth'


def create_ref_vars(inp: InputsManager) -> dict:
    return dict(
        velocity=inp.simulation['reference_velocity'],
        length=inp.simulation['obs_width'],
        force=inp.simulation['obs_mass'] * inp.max_acc,
        angle=math.PI,
        torque=inp.simulation['obs_inertia'] * inp.max_ang_acc,
        time=inp.simulation['obs_width'] / inp.simulation['reference_velocity'],
        ang_velocity=inp.simulation['reference_velocity'] / inp.simulation['obs_width']
    )

def get_env(config_path: str, env_count: int) -> Env:
    def get_env():
        env = TwoWayCouplingConfigEnv(config_path)
        env = StackObservations(env, n_present_features=4, n_past_features=4, past_window=2, append_past_actions=True)
        env = SeedOnResetWrapper(env)
        env = Monitor(env)
        return env

    #venv = DummyVecEnv([get_env for _ in range(4)])
    venv = SubprocVecEnv([get_env for _ in range(env_count)])

    print('Observation space shape: %s' % str(venv.observation_space.shape))
    return venv


def create_model_folder(name: str, storage_base_path: str, config_path: str):
    path_to_model_folder = os.path.join(storage_base_path, name)
    assert os.path.exists(config_path)
    assert os.path.exists(storage_base_path)
    assert not os.path.exists(path_to_model_folder)

    os.mkdir(path_to_model_folder)
    inp = InputsManager(config_path, only=['rl'])
    inp.add_values(os.path.join(inp.rl['simulation_path'], CONFIG_FILENAME), ['simulation'])
    inp.calculate_properties()
    inp.ref_vars = create_ref_vars(inp)
    inp.export(os.path.join(path_to_model_folder, CONFIG_FILENAME))

    return path_to_model_folder

def train_model(path_to_model_folder: str, log_path: str, num_envs: int):
    name = os.path.basename(path_to_model_folder)
    config_path = os.path.join(path_to_model_folder, CONFIG_FILENAME)
    agent_path = os.path.join(path_to_model_folder, AGENT_FILENAME)

    assert os.path.exists(path_to_model_folder)
    assert os.path.exists(config_path)
    assert not os.path.exists(agent_path) # Training continuation currently not supported

    inp = InputsManager(config_path)
    env = get_env(config_path, num_envs)
    agent = SAC('MlpPolicy', env, tensorboard_log=log_path, verbose=1, **inp.rl['training_params'])

    def store_fn(n):
        print('Storing agent...')
        agent.save(agent_path)
        torch_module_path = os.path.join(path_to_model_folder, TORCH_MODEL_FILENAME_TEMPLATE % n)
        # print(torch_module_path)
        store_sac_actor_as_torch_module(agent_path, torch_module_path)

    callback = CallbackList([
        EveryNTimestepsPlusStartFinishFunctionCallback(inp.rl['model_export_stride'], store_fn)
    ])

    agent.learn(total_timesteps=inp.rl['n_iterations'], callback=callback, tb_log_name=name)


if __name__ == '__main__':
    base_directory = os.path.join(os.path.dirname(__file__), os.pardir)
    base_storage_directory = os.path.join(base_directory, 'storage')
    default_config_path = os.path.join(base_directory, 'inputs.json')
    default_model_storage_path = os.path.join(base_storage_directory, 'networks')
    default_log_storage_path = os.path.join(base_storage_directory, 'tensorboard', 'simple_env')

    parser = ArgumentParser()
    parser.add_argument('-n', '--name', dest='name', type=str, help='model name, no storing if not specified')
    parser.add_argument('-c', '--config', dest='config', type=str, default=default_config_path, help='path to config file')
    parser.add_argument('-p', '--path', dest='path', type=str, default=default_model_storage_path, help='path to model storage folder')
    parser.add_argument('-l', '--log', dest='log', type=str, default=default_log_storage_path, help='path to tensorboard logs')
    parser.add_argument('-x', '--num_envs', dest='num_envs', type=int, default=4, help='number of parallel environments for multiprocessing')

    args = parser.parse_args()

    if args.name:
        path_to_model_folder = create_model_folder(args.name, args.path, args.config)
        train_model(path_to_model_folder, args.log, args.num_envs)
    else:
        print('\033[31mNo name specified, training in sandbox mode (no storing)\033[0m')
        inp = InputsManager(args.config)
        env = get_env(args.config, args.num_envs)
        agent = SAC('MlpPolicy', env, verbose=1, **inp.rl['training_params'])
        agent.learn(total_timesteps=inp.rl['n_iterations'])

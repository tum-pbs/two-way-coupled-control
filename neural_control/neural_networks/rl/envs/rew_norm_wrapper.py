from typing import Any, Dict, Optional, Tuple
import numpy as np

from stable_baselines3.common.running_mean_std import RunningMeanStd
from gym import Env, Wrapper


class RewNormWrapper(Wrapper):
    def __init__(self, env: Env, rew_rms: Optional[RunningMeanStd]=None, norm_variance: bool=False):
        super().__init__(env)
        self.unnormalized_ep_rew = None
        self.rew_rms = rew_rms or RunningMeanStd()
        self.norm_variance = norm_variance

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.unnormalized_ep_rew = np.zeros(())
        return obs

    def step(self, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(act)

        self.unnormalized_ep_rew += rew

        if done:
            info['rew_unnormalized'] = self.unnormalized_ep_rew
        
        return obs, self.reward(rew), done, info

    def reward(self, rew: np.ndarray)-> np.ndarray:
        self.rew_rms.update(rew.reshape(-1))
        norm_rew = (rew - self.rew_rms.mean)
        if self.norm_variance:
            norm_rew / np.sqrt(self.rew_rms.var)
        return norm_rew

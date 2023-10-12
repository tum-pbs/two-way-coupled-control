from collections import deque
from typing import Any, Dict, Tuple

import numpy as np
from gym import Env, Wrapper
from gym.spaces import Box


class StackObservations(Wrapper):
    def __init__(self, env: Env, n_present_features: int, n_past_features: int, past_window: int, append_past_actions: bool):
        super().__init__(env)
        self.frames = deque([], maxlen=past_window + 1)
        self.past_actions = deque([], maxlen=past_window)

        self.n_present_features = n_present_features
        self.n_past_obs_features = n_past_features
        self.past_window = past_window
        self.append_past_actions = append_past_actions
        
        self.n_past_features = self.n_past_obs_features
        self.n_action_features = np.prod(env.action_space.shape)
        if self.append_past_actions:
            self.n_past_features += self.n_action_features

        assert isinstance(env.observation_space, Box)
        obs_dim = self.n_present_features + self.n_past_features * self.past_window
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def reset(self) -> np.ndarray:
        obs: np.ndarray = self.env.reset()
        for _ in range(self.past_window):
            self.frames.append(np.zeros(obs.shape))
            self.past_actions.append(np.zeros((self.n_action_features)))
        self.frames.append(obs.copy())
        return self._get_obs()

    def step(self, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        obs, rew, done, info = self.env.step(act)
        self.frames.append(obs.copy())
        self.past_actions.append(act.copy())
        return self._get_obs(), rew, done, info

    def _get_obs(self):
        assert len(self.frames) == self.past_window + 1
        assert len(self.past_actions) == self.past_window

        def collect_obs_features(frame_index: int, feature_count) -> np.ndarray:
            return self.frames[frame_index].reshape(-1)[:feature_count]

        def collect_past_features(frame_index: int) -> np.ndarray:
            obs_features = collect_obs_features(frame_index, self.n_past_obs_features)
            act_features = self.past_actions[frame_index]
            if self.append_past_actions:
                return np.concatenate([obs_features, act_features])
            return obs_features

        return np.concatenate([
            *[collect_past_features(i) for i in range(self.past_window)],
            collect_obs_features(-1, self.n_present_features),
        ])


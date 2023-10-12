from gym import Env, Wrapper
import numpy as np


class SeedOnResetWrapper(Wrapper):
    def __init__(self, env: Env, base_seed: int=0):
        super().__init__(env)
        self._next_seed = base_seed

    def reset(self) -> np.ndarray:
        self.env.seed(self._next_seed)
        self._next_seed += 1
        return self.env.reset()

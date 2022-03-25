import numpy as np
import torch
from typing import Any, Dict, Tuple
from gym import Env, Wrapper


class NumpyWrapper(Wrapper):
    def reset(self)-> np.ndarray:
        return self.env.reset().detach().cpu().numpy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        act_tensor = torch.tensor(action, dtype=torch.float32)
        obs, rew, done, info = self.env.step(act_tensor)
        return obs.detach().cpu().numpy(), rew.detach().cpu().numpy(), done, info

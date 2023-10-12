from typing import Optional
import torch


class SACActorModule(torch.nn.Module):
    def __init__(self, latent_pi: torch.nn.Module, mu: torch.nn.Module, low: float, high: float, obs_shape: tuple):
        super().__init__()

        self.features_extractor = torch.nn.Flatten()
        self.latent_pi = latent_pi
        self.mu = mu
        self.register_buffer('action_space_low', torch.tensor(low))
        self.register_buffer('action_space_high', torch.tensor(high))
        self.obs_shape = obs_shape

    def forward(self, x_present: torch.Tensor, x_past: Optional[torch.Tensor]=None, bypass_tanh: bool=False) -> torch.Tensor:
        x = torch.cat((x_past, x_present), dim=1).to(x_present.device) if x_past is not None else x_present
        return self.rescale(self.mu(self.latent_pi(self.features_extractor(x))), bypass_tanh)

    def rescale(self, x: torch.Tensor, bypass_tanh: bool=False) -> torch.Tensor:
        if not bypass_tanh:
            x = torch.tanh(x)
        return self.action_space_low + (0.5 * (x + 1.0) * (self.action_space_high - self.action_space_low))

    @staticmethod
    def load_from_path(path: str) -> "SACActorModule":
        return torch.load(path)


def store_sac_actor_as_torch_module(agent_path: str, target_path: str):
    from stable_baselines3.sac import SAC

    sb_agent = SAC.load(agent_path)
    sb_actor = sb_agent.policy.actor.cpu()

    th_actor = SACActorModule(sb_actor.latent_pi, sb_actor.mu, sb_agent.action_space.low, sb_agent.action_space.high, sb_agent.observation_space.shape)
    torch.save(th_actor, target_path)


def load_sac_torch_module(path: str) -> SACActorModule:
    return torch.load(path)


if __name__ == '__main__':
    store_sac_actor_as_torch_module('neural_control/storage/networks/sponged_2', '../../../../Documents/GuidedResearch/LOVE2/trained_model_0000.pth')

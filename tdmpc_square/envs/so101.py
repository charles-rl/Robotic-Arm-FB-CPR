import gymnasium as gym
import torch
import numpy as np

# Change this import to match where your actual env class is located!
# If you put so101_env.py inside tdmpc_square/envs/, use:
from tdmpc_square.envs.so101env import RobotArmEnv


class SO101Wrapper(gym.Wrapper):
    def __init__(self, cfg):
        # Parse task name "so101-reach" -> "reach"
        task_parts = cfg.task.split('-')
        task_name = task_parts[1] if len(task_parts) > 1 else "reach"

        env = RobotArmEnv(render_mode=None, reward_type="dense", task=task_name)
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.max_episode_steps = env.max_episode_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(action)

        # obs = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device)
        # reward = torch.tensor([reward], dtype=torch.float32, device=self.cfg.device)
        # done = torch.tensor([terminated or truncated], dtype=torch.bool, device=self.cfg.device)
        return obs, reward, terminated, truncated, info


def make_env(cfg):
    """Factory function."""
    return SO101Wrapper(cfg)
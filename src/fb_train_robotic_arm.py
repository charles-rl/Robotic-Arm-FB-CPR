# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch

# TODO: uncomment when using cuda
# torch.set_float32_matmul_precision("high")

import numpy as np
import dataclasses
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.fb import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from tqdm import tqdm
import time
from dm_control import suite
import random
from pathlib import Path
import wandb
import json
from typing import List
import mujoco
import warnings
import tyro
from generate_dataset import Dataset
from environment import RobotArmEnv
from torch.utils.tensorboard import SummaryWriter

ALL_TASKS = {
    "walker": [
        "walk",
        "run",
        "stand",
    ],
    "cheetah": ["walk", "run"],
    "quadruped": ["walk", "run"],
}


def create_agent(
    observation_dim: int,
    action_dim: int,
    device="cpu",
    compile=False,
    cudagraphs=False,
) -> FBAgentConfig:
    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = observation_dim
    agent_config.model.action_dim = action_dim  # number integer
    agent_config.model.device = device
    agent_config.model.norm_obs = False  # False because observation is normalized
    agent_config.model.seq_length = 1
    agent_config.train.batch_size = 1024
    # archi
    # opted for 50 because I don't need a large latent size
    agent_config.model.archi.z_dim = 50
    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    # divided original dims by 2 because it is overkill
    agent_config.model.archi.b.hidden_dim = 128
    # divided original dims by 8 because it is overkill
    agent_config.model.archi.f.hidden_dim = 256
    agent_config.model.archi.actor.hidden_dim = 256
    # TODO: might consider adding more layers
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2
    # optim default
    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-4
    agent_config.train.lr_actor = 1e-4
    agent_config.train.ortho_coef = 1
    agent_config.train.train_goal_ratio = 0.5
    # changed because fb loss explodes
    agent_config.train.fb_pessimism_penalty = 0.0
    agent_config.train.actor_pessimism_penalty = 0.5

    agent_config.train.discount = 0.98
    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    return agent_config


def load_data(dataset_path):
    env = RobotArmEnv(False)
    dataset_size = torch.load(dataset_path)["counter"]
    data = Dataset(dataset_size, env.observation_space[0], env.action_space[0])
    data.load(dataset_path)
    del env
    print(f"Data path: {dataset_path}")

    limit = data.counter
    storage = {
        "observation": data.observations[:limit].numpy(),
        "action": data.actions[:limit].numpy(),
        # Not used
        "physics": data.arm_raw_angles[:limit].numpy(),
        "next": {
            "observation": data.observations_[:limit].numpy(),
            "terminated": data.dones[:limit].numpy().astype(bool),
            "physics": data.arm_raw_angles[:limit].numpy(),
        }
    }

    # Debug prints
    print("Data loaded successfully.")
    print(f"Obs shape: {storage['observation'].shape}")
    print(f"Action shape: {storage['action'].shape}")
    return storage


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclasses.dataclass
class TrainConfig:
    dataset_root: str = "../data/processed_1000eps.data"
    seed: int = 0
    domain_name: str = "walker"
    task_name: str | None = None
    dataset_expl_agent: str = "rnd"
    num_train_steps: int = 300_000
    load_n_episodes: int = 5_000
    log_every_updates: int = 200
    work_dir: str | None = "../models"
    log_dir: str | None = "../logs"

    checkpoint_every_steps: int = 1_000

    # eval
    num_eval_episodes: int = 3
    num_inference_samples: int = 50_000  # for z sampling
    eval_every_steps: int = 200
    eval_tasks: List[str] | None = None

    # misc
    compile: bool = False
    cudagraphs: bool = False
    # TODO: Turn back to cuda once in server
    # device: str = "cuda"
    device: str = "cpu"

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = "your_wandb_username"
    wandb_gname: str | None = "fb-cpr-robot-arm"
    wandb_pname: str | None = "fb-cpr-robot-arm"
    wandb_name_prefix: str | None = None

    def __post_init__(self):
        self.eval_tasks = ["reach_center", "reach_top", "reach_right_top", "reach_left_top", "reach_bottom"]


class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBAgentConfig) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        if self.cfg.work_dir is None:
            import string

            tmp_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            self.work_dir = Path.cwd() / "tmp_fbcpr" / tmp_name
            self.cfg.work_dir = str(self.work_dir)
        else:
            self.work_dir = Path(self.cfg.work_dir)
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        print(f"working dir: {self.work_dir}")

        self.agent = FBAgent(**dataclasses.asdict(self.agent_cfg))
        set_seed_everywhere(self.cfg.seed)

        if self.cfg.use_wandb:
            exp_name = "fb"
            wandb_name = exp_name
            if self.cfg.wandb_name_prefix:
                wandb_name = f"{self.cfg.wandb_name_prefix}_{exp_name}"
            # fmt: off
            wandb_config = dataclasses.asdict(self.cfg)
            wandb.init(entity=self.cfg.wandb_ename, project=self.cfg.wandb_pname,
                group=self.cfg.agent.name if self.cfg.wandb_gname is None else self.cfg.wandb_gname, name=wandb_name,  # mode="disabled",
                config=wandb_config)  # type: ignore
            # fmt: on

        with (self.work_dir / "config.json").open("w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

        self.writer = SummaryWriter(log_dir=self.cfg.log_dir)

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        self.replay_buffer = {}

        # USE CUSTOM LOADER FOR MY ENVIRONMENT
        data = load_data(self.cfg.dataset_root)

        # Initialize DictBuffer
        self.replay_buffer = {"train": DictBuffer(capacity=data["observation"].shape[0], device=self.agent.device)}
        self.replay_buffer["train"].extend(data)

        # Debug print to verify
        print(self.replay_buffer["train"])

        del data

        total_metrics = None
        fps_start_time = time.time()
        for t in tqdm(range(0, int(self.cfg.num_train_steps))):
            if t % self.cfg.eval_every_steps == 0:
                self.eval(t)

            # torch.compiler.cudagraph_mark_step_begin()
            metrics = self.agent.update(self.replay_buffer, t)

            # we need to copy tensors returned by a cudagraph module
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if t % self.cfg.log_every_updates == 0:
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                for k, v in m_dict.items():
                    if not isinstance(v, str):
                        self.writer.add_scalar(f"train/{k}", v, t)
                total_metrics = None
                fps_start_time = time.time()
            if t % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(str(self.work_dir / "checkpoint"))
        self.agent.save(str(self.work_dir / "checkpoint"))
        return

    def eval(self, t):
        for task in self.cfg.eval_tasks:
            z = self.reward_inference(task).reshape(1, -1)

            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            for ep in range(num_ep):
                eval_env = RobotArmEnv(False)
                done = False
                observation, info = eval_env.reset()
                while not done:
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs = torch.tensor(observation.reshape(1, -1), device=self.agent.device, dtype=torch.float32)
                        action = self.agent.act(obs=obs, z=z, mean=True).cpu().numpy()[0]
                    observation_, reward, terminated, truncated, info = eval_env.step(action, task_name=task)
                    done = terminated or truncated
                    observation = observation_
                    total_reward[ep] += reward
                eval_env.close()
                del eval_env
            m_dict = {
                "reward": np.mean(total_reward),
                "reward#std": np.std(total_reward),
            }
            if self.cfg.use_wandb:
                wandb.log(
                    {f"{task}/{k}": v for k, v in m_dict.items()},
                    step=t,
                )
            m_dict["task"] = task
            print(m_dict)
            for k, v in m_dict.items():
                if not isinstance(v, str):
                    self.writer.add_scalar(f"{m_dict['task']}/{k}", v)

    def reward_inference(self, task_name="testing_task_to_be_replaced") -> torch.Tensor:
        # 1. Sample a batch from the replay buffer
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)

        # Get the next observations (where the arm ended up)
        # Shape: [batch_size, 9] (assuming 3 links * 3 vars)
        next_obs = batch["next"]["observation"]

        # "reach_center", "reach_top", "reach_right_top", "reach_left_top", "reach_bottom"
        if task_name == "reach_center":
            target = torch.tensor([0.0, 0.0], device=self.agent.device)
            gripper_pos = next_obs[:, :2]
            rewards = -torch.norm(gripper_pos - target, dim=1, keepdim=True)
        elif task_name == "reach_top":
            target = torch.tensor([0.0, 2.2], device=self.agent.device)
            gripper_pos = next_obs[:, :2]
            rewards = -torch.norm(gripper_pos - target, dim=1, keepdim=True)
        elif task_name == "reach_right_top":
            target = torch.tensor([1.5, 1.8], device=self.agent.device)
            gripper_pos = next_obs[:, :2]
            rewards = -torch.norm(gripper_pos - target, dim=1, keepdim=True)
        elif task_name == "reach_left_top":
            target = torch.tensor([-1.5, 1.8], device=self.agent.device)
            gripper_pos = next_obs[:, :2]
            rewards = -torch.norm(gripper_pos - target, dim=1, keepdim=True)
        elif task_name == "reach_bottom":
            target = torch.tensor([0.0, -0.5], device=self.agent.device)
            gripper_pos = next_obs[:, :2]
            rewards = -torch.norm(gripper_pos - target, dim=1, keepdim=True)

        rewards = torch.clamp(rewards / 2.5, -1.0, 1.0)
        # 5. Infer z
        # The agent uses the observations and the calculated rewards to guess 'z'
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=rewards.float()
        )
        return z


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    env = RobotArmEnv(False)
    observation_dim = env.observation_space[0]
    action_dim = env.action_space[0]
    del env
    agent_config = create_agent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=config.device,
        compile=config.compile,
        cudagraphs=config.cudagraphs,
    )

    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()

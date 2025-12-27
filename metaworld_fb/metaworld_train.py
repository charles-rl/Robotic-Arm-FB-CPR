# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch

torch.set_float32_matmul_precision("high")

import metaworld
import numpy as np
import dataclasses
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.fb import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode
from tqdm import tqdm
import time
import random
from pathlib import Path
import wandb
import json
from typing import List
import tyro

METAWORLD_TASKS = {
    "reach-v3": 43,
    "push-v3": 40,
    "pick-place-v3": 30,
    "button-press-v3": 6
}

def create_agent(
        device="cuda",
        compile=False,
        cudagraphs=False,
) -> FBAgentConfig:
    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = 36
    agent_config.model.action_dim = 4
    agent_config.model.device = device
    agent_config.model.norm_obs = False
    agent_config.model.seq_length = 1
    agent_config.train.batch_size = 1024
    # archi
    agent_config.model.archi.z_dim = 100
    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.f.hidden_dim = 1024
    agent_config.model.archi.actor.hidden_dim = 1024
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2
    # optim
    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-6
    agent_config.train.lr_actor = 1e-6
    agent_config.train.ortho_coef = 1
    agent_config.train.train_goal_ratio = 0.5
    agent_config.train.fb_pessimism_penalty = 0
    agent_config.train.actor_pessimism_penalty = 0.5

    agent_config.train.discount = 0.99
    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    return agent_config


def load_data(dataset_path, device):
    print(f"Loading Meta-World FB data from: {dataset_path}")
    # Load onto CPU first to avoid GPU memory fragmentation during processing
    data = torch.load(dataset_path, map_location='cpu')

    # Extract arrays
    obs = data['obs']
    actions = data['action']
    rewards = data['reward']
    task_ids = data['task_id']
    episode_ids = data['episode_id']

    # Create sequential pairs (s_t, s_t+1)
    s_t = obs[:-1]
    s_tp1 = obs[1:]
    a_t = actions[:-1]
    r_t = rewards[:-1]
    tid_t = task_ids[:-1]
    ep_t = episode_ids[:-1]
    ep_tp1 = episode_ids[1:]

    # Mask transitions that cross episode boundaries
    valid = (ep_t == ep_tp1)

    # Metamotivo DictBuffer expects everything to be (N, Dim)
    # Even if Dim is 1, it MUST be (N, 1), not (N,)
    storage = {
        "observation": s_t[valid].numpy().astype(np.float32),
        "action": a_t[valid].numpy().astype(np.float32),
        "reward": r_t[valid].numpy().astype(np.float32).reshape(-1, 1),
        "task_id": tid_t[valid].numpy().astype(np.int64).reshape(-1, 1),
        # Add a dummy physics key if the agent's buffer logic expects it from ExORL
        "physics": np.zeros((valid.sum(), 1), dtype=np.float32),
        "next": {
            "observation": s_tp1[valid].numpy().astype(np.float32),
            "terminated": np.zeros((valid.sum(), 1), dtype=np.bool_),
            "physics": np.zeros((valid.sum(), 1), dtype=np.float32),
        },
    }

    print(f"Successfully processed {storage['observation'].shape[0]} transitions.")
    return storage


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclasses.dataclass
class TrainConfig:
    # dataset_root: str = "./fb_metaworld_4tasks_states.pth"
    dataset_root: str = "../../../metaworld_data/parsed_data/fb_metaworld_4tasks_states.pth"
    seed: int = 0
    domain_name: str = "walker"
    task_name: str | None = None
    dataset_expl_agent: str = "rnd"
    num_train_steps: int = 3_000_000
    load_n_episodes: int = 5_000
    log_every_updates: int = 10_000
    work_dir: str | None = None

    checkpoint_every_steps: int = 1_000_000

    # eval
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000
    eval_every_steps: int = 100_000
    eval_tasks: List[str] | None = None

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cuda"

    # WANDB
    use_wandb: bool = True
    wandb_ename: str | None = "charlessosmena0-academia-sinica"
    wandb_gname: str | None = "policy"
    wandb_pname: str | None = "fb_train_metaworld"
    wandb_name_prefix: str | None = None

    def __post_init__(self):
        if self.eval_tasks is None:
            # Evaluate on all 4 tasks by default
            self.eval_tasks = list(METAWORLD_TASKS.keys())


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

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        # Initialize Replay Buffer with the capacity of the filtered dataset
        self.replay_buffer = {}

        # 1. Load the data
        data = load_data(self.cfg.dataset_root, self.agent.device)

        # 2. Initialize Buffer
        # We define capacity based on the dataset size
        capacity = data["observation"].shape[0]

        # Create the buffer
        self.replay_buffer["train"] = DictBuffer(
            capacity=capacity,
            device=self.agent.device
        )

        # 3. Extend the buffer
        # This fills the pre-allocated tensors
        try:
            self.replay_buffer["train"].extend(data)
        except RuntimeError as e:
            print("\n[Buffer Error] Dimension mismatch detected.")
            for k, v in data.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        print(f"Key: next.{k2}, Shape: {v2.shape}")
                else:
                    print(f"Key: {k}, Shape: {v.shape}")
            raise e

        print(f"Replay Buffer Ready: {capacity} transitions.")
        del data  # Clean up CPU memory

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
                total_metrics = None
                fps_start_time = time.time()
            if t % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(str(self.work_dir / "checkpoint"))
        self.agent.save(str(self.work_dir / "checkpoint"))
        return

    def eval(self, t):
        # Define the mapping we found earlier
        task_to_id = {
            "reach-v3": 43,
            "push-v3": 40,
            "pick-place-v3": 30,
            "button-press-v3": 6
        }

        for task in self.cfg.eval_tasks:
            # Step 1: Get task-specific Z using sampled task-specific data
            task_id = task_to_id.get(task)
            z = self.reward_inference(task, task_id).reshape(1, -1)

            # Step 2: Initialize Meta-World Env
            mt1 = metaworld.MT1(task)
            eval_env = mt1.train_classes[task]()
            eval_env.set_task(random.choice(mt1.train_tasks))

            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            for ep in range(num_ep):
                # Gymnasium reset returns (obs, info)
                obs, _ = eval_env.reset()
                done = False

                while not done:
                    with torch.no_grad(), eval_mode(self.agent._model):
                        # Mask Goal: Only first 36 dims for state s
                        state = torch.tensor(
                            obs[:36].reshape(1, -1),
                            device=self.agent.device,
                            dtype=torch.float32,
                        )
                        action = self.agent.act(obs=state, z=z, mean=True).cpu().numpy().flatten()

                    # Gymnasium step returns 5 values
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    total_reward[ep] += reward
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

    def reward_inference(self, task, task_id) -> torch.Tensor:
        """
        Modified to use task-specific samples from the replay buffer
        instead of simulator stepping.
        """
        num_samples = self.cfg.num_inference_samples

        # 1. Filter buffer for transitions belonging to this specific task_id
        # Note: self.replay_buffer['train']._storage access depends on your Buffer implementation
        all_task_ids = self.replay_buffer["train"].storage["task_id"].flatten()
        task_indices = np.where(all_task_ids == task_id)[0]

        if len(task_indices) < num_samples:
            # Fallback if we have fewer samples than requested
            idx = task_indices
        else:
            idx = np.random.choice(task_indices, num_samples, replace=False)

        # 2. Extract batch from these indices
        batch_next_obs = self.replay_buffer["train"].storage["next"]["observation"][idx]
        batch_rewards = self.replay_buffer["train"].storage["reward"][idx]

        # Convert to Tensors
        next_obs = torch.tensor(batch_next_obs, device=self.agent.device, dtype=torch.float32).detach().clone()
        rewards = torch.tensor(batch_rewards, device=self.agent.device, dtype=torch.float32).detach().clone()

        # 3. Solve for z
        z = self.agent._model.reward_inference(
            next_obs=next_obs,
            reward=rewards
        )
        return z


if __name__ == "__main__":
    # Use tyro to parse TrainConfig
    config = tyro.cli(TrainConfig)

    # Initialize FB Agent Config for Meta-World
    agent_config = create_agent(
        device=config.device,
        compile=config.compile,
        cudagraphs=config.cudagraphs,
    )

    # Initialize Workspace and Start Training
    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree

from __future__ import annotations
import torch

EVAL = False
DEBUG = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.set_float32_matmul_precision("high")

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
import imageio

# Import your environment classes to get dimensions later
from environment import SO101LiftEnv, SO101ReachEnv

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
    agent_config.model.norm_obs = True
    agent_config.model.seq_length = 1
    agent_config.train.batch_size = 512
    # archi
    agent_config.model.archi.z_dim = 30
    agent_config.model.archi.b.norm = True
    agent_config.model.archi.norm_z = True
    agent_config.model.archi.b.hidden_dim = 256
    agent_config.model.archi.f.hidden_dim = 1024
    agent_config.model.archi.actor.hidden_dim = 1024
    agent_config.model.archi.f.hidden_layers = 1
    agent_config.model.archi.actor.hidden_layers = 1
    agent_config.model.archi.b.hidden_layers = 2
    # optim default
    agent_config.train.lr_f = 1e-4
    agent_config.train.lr_b = 1e-6
    agent_config.train.lr_actor = 1e-6
    agent_config.train.ortho_coef = 1
    agent_config.train.train_goal_ratio = 0.5

    # changed because fb loss explodes
    agent_config.train.fb_pessimism_penalty = 0.0
    agent_config.train.actor_pessimism_penalty = 0.5

    agent_config.train.discount = 0.9
    agent_config.compile = compile
    agent_config.cudagraphs = cudagraphs

    return agent_config


def load_data(dataset_path):
    print(f"🔄 Loading data from: {dataset_path}")
    raw_data = np.load(dataset_path)
    print(f"Data path: {dataset_path}")

    def to_2d(arr):
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    reward_scale = 1.0 / 8.0 # divided by 8.0 since 8 is max for lift task
    # TODO: this scale is only for lift task

    storage = {
        "observation": raw_data["observation"][:-1].astype(np.float32),
        "action": raw_data["action"][:-1].astype(np.float32),
        "physics": raw_data["physics"][:-1].astype(np.float32),

        # --- NEW FIELDS LOADED HERE ---
        # These are stored in the buffer and will be available in batch samples
        "raw_motor_ctrl": raw_data["raw_motor_ctrl"][:-1].astype(np.float32),
        "task_ids": to_2d(raw_data["task_ids"][:-1]).astype(np.int32),
        "episode_ids": to_2d(raw_data["episode_ids"][:-1]).astype(np.int32),
        "reward": to_2d(raw_data["reward"][:-1] * reward_scale).astype(np.float32),
        # -----------------------------
        "next": {
            "reward": to_2d(raw_data["reward"][1:] * reward_scale).astype(np.float32),
            "observation": raw_data["observation"][1:].astype(np.float32),
            "physics": raw_data["physics"][1:].astype(np.float32),
            "terminated": to_2d(raw_data["terminated"][:-1]).astype(bool),
        }
    }

    # Debug prints
    print("✅ Data loaded successfully.")
    print(f"   Obs shape: {storage['observation'].shape}")
    print(f"   Action shape: {storage['action'].shape}")
    print(f"   Extra fields loaded: raw_motor_ctrl, task_ids, episode_ids, reward")
    return storage


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclasses.dataclass
class TrainConfig:
    dataset_root: str = "../data/lift_merged.npz"
    seed: int = 0
    domain_name: str = "walker"
    task_name: str | None = None
    dataset_expl_agent: str = "rnd"
    num_train_steps: int = 800_000
    load_n_episodes: int = 5_000
    log_every_updates: int = 1000
    work_dir: str | None = "../models"
    log_dir: str | None = "../logs"

    checkpoint_every_steps: int = 100_000

    # eval
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000  # for z sampling
    eval_every_steps: int = 10_000
    eval_tasks: List[str] | None = None

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = DEVICE

    # WANDB
    use_wandb: bool = True
    wandb_ename: str | None = "charlessosmena0-academia-sinica"
    wandb_gname: str | None = "fb"
    wandb_pname: str | None = "so101-lift-fb"
    wandb_name_prefix: str | None = None

    def __post_init__(self):
        self.eval_tasks = ["lift"]


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

        # my changes
        self.best_avg_reward = float("-inf")

    def init_replay_buffer(self):
        """
        Loads the merged dataset and initializes the FB Replay Buffer.
        """
        print("⏳ Initializing Replay Buffer...")
        self.replay_buffer = {}

        # 1. Load Data using the custom load_data function we defined
        # This handles the slicing: 800,000 -> 799,999 valid transitions
        data = load_data(self.cfg.dataset_root)
        dataset_size = data["observation"].shape[0]

        # 2. Create the DictBuffer
        self.replay_buffer = {
            "train": DictBuffer(
                capacity=dataset_size,
                device=self.agent.device
            )
        }

        self.replay_buffer["train"].extend(data)

        del data

        print(f"✅ Buffer ready with {dataset_size} transitions.")

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        self.init_replay_buffer()
        total_metrics = None
        fps_start_time = time.time()
        print(f"\n🚀 Starting training for {self.cfg.num_train_steps} steps...")
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
        self.eval(self.cfg.num_train_steps)
        self.agent.save(str(self.work_dir / "checkpoint"))
        return

    def eval(self, t):
        all_tasks_total_reward = np.zeros((len(self.cfg.eval_tasks),), dtype=np.float64)
        for task_idx, task in enumerate(self.cfg.eval_tasks):
            z = self.reward_inference(task).reshape(1, -1)

            render_mode = "rgb_array"
            if "lift" in task:
                eval_env = SO101LiftEnv(
                    render_mode=render_mode,
                    reward_type="dense",
                    control_mode="delta_end_effector",
                    fb_train=True,
                    evaluate=True,
                )

            num_ep = self.cfg.num_eval_episodes
            total_reward = np.zeros((num_ep,), dtype=np.float64)
            for ep in range(num_ep):
                observation, _ = eval_env.reset()
                ep_frames = []
                done = False
                while not done:
                    with torch.no_grad(), eval_mode(self.agent._model):
                        obs = torch.tensor(observation, device=self.agent.device, dtype=torch.float32).reshape(1, -1)
                        action = self.agent.act(obs=obs, z=z, mean=True).cpu().numpy()[0]
                    observation_, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    observation = observation_
                    total_reward[ep] += reward

                    if EVAL:
                        frame = eval_env.render()
                        ep_frames.append(frame)

                    if ep == 0:
                        frames = ep_frames

                eval_env.close()

            m_dict = {
                "reward": np.mean(total_reward),
                "reward#std": np.std(total_reward),
            }
            all_tasks_total_reward[task_idx] = m_dict["reward"]
            if self.cfg.use_wandb:
                wandb.log(
                    {f"{task}/{k}": v for k, v in m_dict.items()},
                    step=t,
                )
                #TODO: wrong logged value here, for some reason it's just 0

            m_dict["task"] = task
            print(m_dict)
        all_tasks_avg_reward = np.mean(all_tasks_total_reward)
        if all_tasks_avg_reward > self.best_avg_reward:
            self.best_avg_reward = all_tasks_avg_reward
            self.agent.save(str(self.work_dir / "best checkpoint"))
            print(f"🌟 New Best Model! Avg Reward: {all_tasks_avg_reward:.2f} (Was: {self.best_avg_reward:.2f})")

        if EVAL:
            video_name = f"{task}_eval{t}.mp4"
            imageio.mimsave(str(self.work_dir / video_name), frames, fps=30)
            print(f"   💾 Saved video to {video_name}")

    def reward_inference(self, task_name) -> torch.Tensor:
        # 1. Sample a batch
        num_samples = self.cfg.num_inference_samples
        batch = self.replay_buffer["train"].sample(num_samples)

        # TODO: In the future given task name then sample only the rewards with those task ids
        # This is only a batch sample so in the future it must be batch sampling that specific task ID

        # 6. Infer z
        # The network expects NORMALIZED obs, so we pass the original batch
        z = self.agent._model.reward_inference(
            next_obs=batch["next"]["observation"],
            reward=batch["next"]["reward"],
        )
        return z


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)

    # fb_train=True to get fb observation space
    env = SO101ReachEnv(fb_train=True, evaluate=True, control_mode="delta_end_effector")
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    del env
    agent_config = create_agent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        device=config.device,
        compile=config.compile,
        cudagraphs=config.cudagraphs,
    )

    if DEBUG or EVAL:
        config.log_every_updates = 100
        config.eval_every_steps = 100
        config.num_eval_episodes = 1
        config.use_wandb = False

    ws = Workspace(config, agent_cfg=agent_config)
    if not EVAL:
        ws.train()

    if EVAL:
        # print("Starting  Evaluation...")
        # ws.init_replay_buffer()
        # ws.agent = FBAgent.load(str(ws.work_dir / "server/checkpoint"), device=ws.cfg.device)
        # # ws.agent.load(str(ws.work_dir / "checkpoint"), device=ws.cfg.device)
        # ws.eval(0)

        print("Starting Best Evaluation...")
        ws.init_replay_buffer()
        ws.agent = FBAgent.load(str(ws.work_dir / "server/best checkpoint"), device=ws.cfg.device)
        # ws.agent.load(str(ws.work_dir / "checkpoint"), device=ws.cfg.device)
        # while True:
        #     ws.eval(0)
        #     if ws.best_avg_reward > 400.0:
        #         break
        ws.eval(0)
        ws.eval(1)
        ws.eval(2)

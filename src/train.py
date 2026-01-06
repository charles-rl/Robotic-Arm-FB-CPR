import os
import imageio
import wandb
import torch
import numpy as np

from sb3_contrib import TQC, CrossQ
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

from environment import SO101LiftEnv, SO101ReachEnv

# ==================================================================================
#                                 CONFIGURATION
# ==================================================================================
CONFIG = {
    "DEBUG": False,
    "EVAL": False,  # Set to False to Train, True to Evaluate
    "TASK": "reach",  # "lift" or "reach"
    "FORCED_CUBE_POSITION": -1,  # 0 for Center, 1 for Left, 2 for Right, 3 for Far Left, 4 for Far Right
    "ALGO": "TQC",  # "SAC", "TQC", "PPO", "CrossQ"
    "CONTROL_MODE": "delta_end_effector",  # "delta_end_effector" or "delta_joint_position"
    "REWARD_THRESHOLD": 100.0,

    # Checkpointing
    "LOAD_CHECKPOINT": False,
    "CHECKPOINT_DIR": "../models",
    "LOG_DIR": "./models",
    "DATASET_DIR": "../data",

    # Training Hyperparameters (Production)
    "TOTAL_TIMESTEPS": 1_500_000,
    "BATCH_SIZE": 512,
    "BUFFER_SIZE": 1_000_000,
    "LOG_INTERVAL": 10,
    "SAVE_FREQ": 10_000,
    "EVAL_FREQ": 2_000,
    "DATASET_SAVE_FREQ": 50_000,

    # Training Hyperparameters (Debug)
    "DEBUG_TIMESTEPS": 2_000,
    "DEBUG_BATCH_SIZE": 64,
    "DEBUG_BUFFER_SIZE": 5_000,
    "DEBUG_DATASET_SAVE_FREQ": 200,
}

CONFIG["RUN_NAME"] = f"{CONFIG['ALGO']}_{CONFIG['CONTROL_MODE']}_{CONFIG['TASK']}"
CONFIG["MODEL_PATH"] = os.path.join(CONFIG["CHECKPOINT_DIR"], f"{CONFIG['ALGO'].lower()}_so101_{CONFIG['TASK']}.zip")
CONFIG["STATS_PATH"] = os.path.join(CONFIG["CHECKPOINT_DIR"], f"vec_normalize_{CONFIG['TASK']}.pkl")


# ==================================================================================
#                                    CALLBACKS
# ==================================================================================

class RawRewardCallback(BaseCallback):
    """Logs raw reward/length to WandB."""

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info:
                wandb.log({
                    "rollout/raw_episode_reward": info['episode']['r'],
                    "rollout/raw_episode_length": info['episode']['l'],
                    "env_steps": self.num_timesteps
                })
        return True


class DataCollectorCallback(BaseCallback):
    """
    Collects raw data (including info dicts) for FB representation learning.
    Saves to .npz files.
    """

    def __init__(self, save_dir, task_name, save_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.task_name = task_name
        self.save_freq = save_freq
        self.buffer = {
            "observation": [],  # The 'fb_obs' from info (unnormalized)
            "action": [],  # The action taken
            "reward": [],  # The reward received
            "terminated": [],  # Done flag
            "physics": [],  # qpos + qvel
            "raw_motor_ctrl": [],  # 6-DOF motor command
            "task_ids": [],
            "episode_ids": [],
            "cube_focus_idxs": [],
            "cube_pos_idxs": [],
        }
        self.chunk_id = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Retrieve data from the vectorized environment
        # Note: self.locals['infos'] is a list of dicts (one per env)
        # We assume DummyVecEnv with 1 env for simplicity, but we iterate to be safe.

        infos = self.locals['infos']
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        dones = self.locals['dones']

        for i, info in enumerate(infos):
            # 1. Get Physics & Motor (Crucial for FB)
            if "physics" in info:
                self.buffer["physics"].append(info["physics"])
            else:
                # Fallback if info missing (shouldn't happen with your env)
                pass

            if "raw_motor_ctrl" in info:
                self.buffer["raw_motor_ctrl"].append(info["raw_motor_ctrl"])

            # 2. Get Observation
            # We prefer 'fb_obs' from info because it is UNNORMALIZED.
            # VecNormalize would obscure the real state.
            if "fb_obs" in info:
                self.buffer["observation"].append(info["fb_obs"])
            else:
                # Fallback to the normalized obs if fb_obs missing
                # self.locals['new_obs'][i]
                pass

            self.buffer["episode_ids"].append(info["episode_id"])
            self.buffer["task_ids"].append(info["task_id"])
            self.buffer["cube_focus_idxs"].append(info["cube_focus_idx"])
            self.buffer["cube_pos_idxs"].append(info["cube_pos_idx"])


            # 3. Standard RL data
            self.buffer["action"].append(actions[i])
            self.buffer["reward"].append(rewards[i])
            self.buffer["terminated"].append(dones[i])

        # Periodically dump to disk to avoid OOM on massive runs
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            self._save_chunk()

        return True

    def _save_chunk(self):
        if len(self.buffer["observation"]) == 0:
            return

        filename = os.path.join(self.save_dir, f"{self.task_name}_chunk_{self.chunk_id}.npz")

        # Convert list to numpy arrays
        data_to_save = {k: np.array(v) for k, v in self.buffer.items()}

        np.savez_compressed(filename, **data_to_save)

        if self.verbose > 0:
            print(f"💾 Saved data chunk {self.chunk_id} to {filename} ({len(self.buffer['observation'])} steps)")

        # Clear memory
        self.chunk_id += 1
        for k in self.buffer:
            self.buffer[k] = []

    def _on_training_end(self) -> None:
        # Save remaining data
        self._save_chunk()
        print("✅ Data Collection Complete.")


class CheckpointWithStatsCallback(BaseCallback):
    """Saves the model AND the VecNormalize statistics."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 1. Save Model
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(model_path)
            # 2. Save Stats
            stats_path = os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl")
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save(stats_path)
                if self.verbose > 1:
                    print(f"Saved model and stats to {self.save_path}")
            else:
                if self.verbose > 0:
                    print("⚠️ Warning: VecNormalize not found, only saved model.")
        return True


# ==================================================================================
#                                 HELPER FUNCTIONS
# ==================================================================================

def make_env_class(task_name):
    """Returns the class constructor based on task string."""
    if task_name == "lift":
        return SO101LiftEnv
    elif task_name == "reach":
        return SO101ReachEnv
    else:
        raise ValueError(f"Unknown task: {task_name}")


def get_model_class(algo_name):
    if algo_name == "TQC":
        return TQC
    elif algo_name == "SAC":
        return SAC
    elif algo_name == "PPO":
        return PPO
    elif algo_name == "CrossQ":
        return CrossQ
    raise ValueError(f"Unknown Algorithm: {algo_name}")


def create_vec_env(task, control_mode, evaluate=False, stats_path=None, training=True):
    """Creates a normalized vectorized environment."""
    EnvClass = make_env_class(task)

    # 1. Create Base Env
    # Note: Monitor wrapper is crucial for recording episode stats
    env = DummyVecEnv([lambda: Monitor(EnvClass(
        render_mode=None,  # Training is headless
        reward_type="dense",
        control_mode=control_mode,
        evaluate=evaluate,
        forced_cube_pos_idx=CONFIG["FORCED_CUBE_POSITION"]
    ))])

    # 2. Add Normalization
    if stats_path and os.path.exists(stats_path):
        print(f"🔄 Loading Env Stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = training  # Update stats if training, freeze if eval
        env.norm_reward = False  # Usually keep rewards raw for clarity
    else:
        print("✨ Creating New Env Stats")
        # clip_obs=10.0 is standard stable-baselines3 default
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., training=training)

    return env


# ==================================================================================
#                                   MAIN LOGIC
# ==================================================================================

def test_environment_structure():
    print("--- 1. Checking Environment Compliance ---")
    EnvClass = make_env_class(CONFIG["TASK"])
    env = EnvClass(reward_type="dense", control_mode=CONFIG["CONTROL_MODE"])
    try:
        check_env(env)
        print(f"✅ {CONFIG['TASK']} Environment passed Gymnasium checks!")
    except Exception as e:
        print(f"❌ {CONFIG['TASK']} Environment failed checks: {e}")
    env.close()


def train_agent():
    print(f"\n--- 2. Setting up Training ({CONFIG['ALGO']}) for task: {CONFIG['TASK']} ---")

    # Setup Hyperparams based on DEBUG flag
    total_timesteps = CONFIG["DEBUG_TIMESTEPS"] if CONFIG["DEBUG"] else CONFIG["TOTAL_TIMESTEPS"]
    batch_size = CONFIG["DEBUG_BATCH_SIZE"] if CONFIG["DEBUG"] else CONFIG["BATCH_SIZE"]
    buffer_size = CONFIG["DEBUG_BUFFER_SIZE"] if CONFIG["DEBUG"] else CONFIG["BUFFER_SIZE"]
    log_interval = 1 if CONFIG["DEBUG"] else CONFIG["LOG_INTERVAL"]
    dataset_save_freq = CONFIG["DEBUG_DATASET_SAVE_FREQ"] if CONFIG["DEBUG"] else CONFIG["DATASET_SAVE_FREQ"]

    # 1. WandB Init
    run = wandb.init(
        project=f"so101-{CONFIG['TASK']}-fb",
        name=CONFIG["RUN_NAME"],
        config=CONFIG,
        sync_tensorboard=True,
    )

    # 2. Environment Setup
    # Load stats if checkpoint exists, otherwise create new
    stats_load_path = CONFIG["STATS_PATH"] if CONFIG["LOAD_CHECKPOINT"] else None
    env = create_vec_env(CONFIG["TASK"], CONFIG["CONTROL_MODE"], evaluate=False, stats_path=stats_load_path,
                         training=True)

    # 3. Model Setup
    ModelClass = get_model_class(CONFIG["ALGO"])

    if CONFIG["LOAD_CHECKPOINT"] and os.path.exists(CONFIG["MODEL_PATH"]):
        print(f"🔄 Loading Model Weights from {CONFIG['MODEL_PATH']}")
        model = ModelClass.load(
            CONFIG["MODEL_PATH"],
            env=env,
            tensorboard_log=f"runs/{run.id}",
            custom_objects={
                "learning_rate": 1e-5,  # Fine-tuning LR
                "ent_coef": "auto_0.1"
            }
        )
    else:
        print("✨ Initializing New Model")
        policy_kwargs = dict(net_arch=[512, 512])

        # Algo Specific Configs
        if CONFIG["ALGO"] == "TQC":
            policy_kwargs["use_sde"] = True
            policy_kwargs["log_std_init"] = -2
            model = TQC(
                "MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}",
                policy_kwargs=policy_kwargs,
                top_quantiles_to_drop_per_net=2,
                learning_rate=3e-4,
                gamma=0.99,
                batch_size=batch_size,
                buffer_size=buffer_size,
                train_freq=(200, "step"),
                gradient_steps=200,
                ent_coef="auto"
            )
        elif CONFIG["ALGO"] == "CrossQ":
            model = CrossQ(
                "MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}",
                learning_rate=1e-3,
                batch_size=1024,  # CrossQ needs large batch
                buffer_size=buffer_size,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=policy_kwargs,
                ent_coef="auto",
                target_entropy=-3.5
            )
        else:
            # Fallback for SAC/PPO default params
            model = ModelClass("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}",
                               policy_kwargs=policy_kwargs)

    # 4. Callbacks
    # Eval Env (Independent stats, freeze training)
    eval_env = create_vec_env(CONFIG["TASK"], CONFIG["CONTROL_MODE"], evaluate=True, stats_path=stats_load_path,
                              training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{CONFIG['LOG_DIR']}/{run.id}/best_model",
        log_path=f"{CONFIG['LOG_DIR']}/{run.id}/eval_logs",
        eval_freq=CONFIG["EVAL_FREQ"],
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointWithStatsCallback(
        save_freq=CONFIG["SAVE_FREQ"],
        save_path=f"{CONFIG['LOG_DIR']}/{run.id}",
        name_prefix="rl_model"
    )

    data_collector = DataCollectorCallback(
        save_dir=os.path.join(CONFIG["DATASET_DIR"], CONFIG["TASK"]),
        task_name=CONFIG["TASK"],
        save_freq=dataset_save_freq,  # e.g. every 100k steps
        verbose=1
    )

    callback_group = CallbackList([
        RawRewardCallback(),
        WandbCallback(gradient_save_freq=200, verbose=2),
        checkpoint_callback,
        eval_callback,
        data_collector
    ])

    print("--- 3. Starting Training ---")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        log_interval=log_interval,
        callback=callback_group,
        reset_num_timesteps=True
    )

    # 5. Final Save
    os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
    model.save(CONFIG["MODEL_PATH"])
    env.save(CONFIG["STATS_PATH"])
    print(f"✅ Final Model saved to {CONFIG['MODEL_PATH']}")

    run.finish()
    env.close()


def evaluate(episodes=3, evaluate_type=True, only_visualize=False):
    print(f"\n--- 4. Evaluating Agent: {CONFIG['TASK']} ({CONFIG['ALGO']}) - Mode: {evaluate_type} ---")

    EnvClass = make_env_class(CONFIG["TASK"])
    render_mode = "human" if only_visualize else "rgb_array"

    # 1. Create Eval Env
    # Note: We use DummyVecEnv manually here to control rendering loop
    def make_eval_env():
        return EnvClass(render_mode=render_mode, reward_type="dense", control_mode=CONFIG["CONTROL_MODE"],
                        evaluate=evaluate_type, forced_cube_pos_idx=CONFIG["FORCED_CUBE_POSITION"])

    env_ = DummyVecEnv([make_eval_env])
    max_steps = env_.get_attr("max_episode_steps")[0]

    # 2. Load Stats
    if os.path.exists(CONFIG["STATS_PATH"]):
        env = VecNormalize.load(CONFIG["STATS_PATH"], env_)
        env.training = False
        env.norm_reward = False
        print("Loaded Normalization Stats.")
    else:
        print("⚠️ Warning: No normalization stats found. Evaluation might be suboptimal.")
        env = env_

    # 3. Load Model
    ModelClass = get_model_class(CONFIG["ALGO"])
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        print(f"❌ Model not found at {CONFIG['MODEL_PATH']}")
        return

    model = ModelClass.load(CONFIG["MODEL_PATH"])

    # 4. Simulation Loop
    for ep in range(episodes):
        frames = []
        obs = env.reset()
        desc_str = str(evaluate_type) if isinstance(evaluate_type, str) else "eval"
        video_path = f"{CONFIG['TASK']}_{desc_str}_{ep}.mp4"
        print(f"Simulating Episode {ep + 1}/{episodes}...")

        total_reward = 0
        for _ in range(max_steps + 10):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            total_reward += rewards[0]

            if only_visualize:
                env.envs[0].render()
            else:
                frame = env.envs[0].render()
                frames.append(frame)

            if dones[0]: break

        print(f"Episode {ep + 1} Reward: {total_reward:.2f}")

        if not only_visualize:
            print(f"Saving {video_path}...")
            imageio.mimsave(video_path, frames, fps=30)

    env.close()


if __name__ == "__main__":
    test_environment_structure()

    if not CONFIG["EVAL"]:
        train_agent()
    else:
        # Run standard evaluation suite
        if CONFIG["TASK"] == "lift":
            evaluate(episodes=1, evaluate_type="hold")
            evaluate(episodes=1, evaluate_type="hoist")
            evaluate(episodes=1, evaluate_type="prehoist")
            evaluate(episodes=3, evaluate_type=True)  # Full random eval
        elif CONFIG["TASK"] == "reach":
            evaluate(episodes=5, evaluate_type=True)  # Full random eval
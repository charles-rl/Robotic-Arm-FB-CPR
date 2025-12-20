import numpy as np
import gymnasium
import mujoco
import mujoco.viewer
import imageio
import wandb
import os

from sb3_contrib import TQC, CrossQ
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

from src.environment import RobotArmEnv

# Configuration
DEBUG = True
EVAL = True
TASK = "lift"
ALGO = "TQC"  # <--- CHANGE THIS: "SAC", "TQC", or "PPO" or "CrossQ"
CONTROL_MODE = "delta_end_effector"  # delta_end_effector delta_joint_position
REWARD_THRESHOLD = 100.0
RUN_NAME = f"{ALGO}_{CONTROL_MODE}"

def test_consistency(model_path, stats_path, n_episodes=50):
    print(f"\n--- 5. Consistency Check ({n_episodes} eps per stage) ---")

    # 1. Setup Env (No rendering for speed)
    env_ = DummyVecEnv(
        [lambda: RobotArmEnv(render_mode=None, reward_type="dense", task="lift", control_mode=CONTROL_MODE)])

    # Load Stats
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env_)
        env.training = False
        env.norm_reward = False
    else:
        env = env_

    # Load Model
    # (Assuming TQC for now based on your previous success)
    model = TQC.load(model_path)

    # 2. Define Stages to Test
    # We "hack" the env to force specific reset modes if possible,
    # but since your env randomizes internally, we just run many episodes
    # and infer the stage from the starting reward/height.

    results = {
        "hold_stage": {"success": 0, "total": 0, "rewards": []},
        "hoist_stage": {"success": 0, "total": 0, "rewards": []},
        "random_stage": {"success": 0, "total": 0, "rewards": []}
    }

    print(f"Running {n_episodes * 3} episodes total...")

    for i in range(n_episodes * 3):
        obs = env.reset()

        # HACK: Infer stage from initial observation
        # Get start Z height of the target cube (you might need to expose this in info)
        # For now, let's look at the "EE-Cube Relative Z" or similar if available
        # Or just rely on the law of large numbers.

        # Better way: Access the internal env directly
        internal_env = env.envs[0].unwrapped
        # Check internal state variables we set in reset()
        # This requires the env to expose which stage it picked.
        # Let's assume you add `self.current_stage = "hold"` in your reset()

        # If you haven't added that flag yet, we can infer from Z height
        cube_z = internal_env.data.xpos[internal_env.cube_a_id][2]  # Simplified access
        if cube_z > 0.5 + internal_env.base_pos_world[2]:
            stage = "hold_stage"
        elif cube_z > 0.05 + internal_env.base_pos_world[2]:
            stage = "hoist_stage"  # On table
        else:
            stage = "random_stage"  # Likely random

        episode_reward = 0
        final_cube_z = 0

        for _ in range(200):  # Max steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            if done[0]: break

        # Check Success Condition (Lifted > 15cm)
        # We need to get the final state from the env
        final_cube_z = internal_env.data.xpos[internal_env.cube_a_id][2]
        is_success = final_cube_z > 0.15

        # Log
        results[stage]["total"] += 1
        results[stage]["rewards"].append(episode_reward)
        if is_success: results[stage]["success"] += 1

        print(f"\rProgress: {i + 1}/{n_episodes * 3}", end="")

    print("\n\n=== CONSISTENCY REPORT ===")
    for stage, data in results.items():
        if data["total"] == 0: continue
        sr = (data["success"] / data["total"]) * 100
        avg_r = np.mean(data["rewards"])
        std_r = np.std(data["rewards"])
        print(f"[{stage.upper()}]")
        print(f"  Success Rate: {sr:.1f}%")
        print(f"  Avg Reward:   {avg_r:.1f} ± {std_r:.1f}")
        print(f"  (Drift Check: High Std Dev = Unstable Hold)")
        print("-" * 20)


if __name__ == "__main__":
    # Add this line to run it
    test_consistency(
        model_path=f"../models/{ALGO.lower()}_so101_{TASK}",
        stats_path=f"../models/vec_normalize_{TASK}.pkl",
        n_episodes=50
    )

import os
import pickle
import numpy as np
import random
import metaworld
from metaworld.policies import (
    SawyerReachV3Policy,
    SawyerPushV3Policy,
    SawyerPickPlaceV3Policy
)
from tqdm import tqdm


class FBDataGenerator:
    def __init__(self, tasks, episodes_per_task=100):
        self.task_names = tasks
        self.ep_per_task = episodes_per_task
        self.dataset = []

        # Map task names to their respective scripted policies
        self.policy_map = {
            'reach-v3': SawyerReachV3Policy,
            'push-v3': SawyerPushV3Policy,
            'pick-place-v3': SawyerPickPlaceV3Policy
        }

    def get_physics_state(self, env):
        """Extracts qpos and qvel to match MuJoCo set_state requirements."""
        # For MuJoCo 2.0+ (used by recent Meta-World versions)
        # qpos: generalized positions, qvel: generalized velocities
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def collect_data(self, save_path="metaworld_fb_dataset.pkl"):
        for task_name in self.task_names:
            print(f"Collecting data for: {task_name}")

            mt1 = metaworld.MT1(task_name)
            env = mt1.train_classes[task_name]()
            policy = self.policy_map[task_name]()

            for ep in tqdm(range(self.ep_per_task)):
                task = np.random.choice(mt1.train_tasks)
                env.set_task(task)

                obs, _ = env.reset()
                # Capture physics state after reset
                current_physics = self.get_physics_state(env)

                done = False
                step = 0

                while not done and step < env.max_path_length:
                    # Scripted policy needs the full 39-D obs
                    action = policy.get_action(obs)

                    # Step the environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    next_physics = self.get_physics_state(env)

                    done = terminated or truncated

                    # FB Separation:
                    # state: 0-35 (robot + objects)
                    # goal: 36-38 (target)
                    state_only = obs[:36]
                    next_state_only = next_obs[:36]
                    goal_only = obs[36:39]

                    self.dataset.append({
                        'observation': state_only.astype(np.float32),
                        'action': action.astype(np.float32),
                        'next_observation': next_state_only.astype(np.float32),
                        'reward': reward,
                        'terminal': terminated,
                        'physics': current_physics,  # qpos + qvel before step
                        'next_physics': next_physics,  # qpos + qvel after step
                        'goal': goal_only.astype(np.float32),
                        'task_id': task_name
                    })

                    obs = next_obs
                    current_physics = next_physics
                    step += 1

        self._save(save_path)

    def _save(self, path):
        print(f"Saving {len(self.dataset)} transitions to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f)
        print("Done.")


if __name__ == "__main__":
    # You can increase episodes_per_task based on your storage/memory
    generator = FBDataGenerator(
        tasks=['reach-v3', 'push-v3', 'pick-place-v3'],
        episodes_per_task=500
    )
    generator.collect_data()

import os
import pickle
import numpy as np
import metaworld
from metaworld.policies import (
    SawyerReachV3Policy,
    SawyerPushV3Policy,
    SawyerPickPlaceV3Policy
)
from tqdm import tqdm


class FBDataGenerator:
    def __init__(self, tasks=['reach-v2', 'push-v2', 'pick-place-v2'], episodes_per_task=100):
        self.task_names = tasks
        self.ep_per_task = episodes_per_task
        self.dataset = []

        # Map task names to their respective scripted policies
        self.policy_map = {
            'reach-v2': SawyerReachV3Policy,
            'push-v2': SawyerPushV3Policy,
            'pick-place-v2': SawyerPickPlaceV3Policy
        }

    def collect_data(self, save_path="metaworld_fb_dataset.pkl"):
        for task_name in self.task_names:
            print(f"Collecting data for: {task_name}")

            # 1. Initialize MT1 for the specific task
            mt1 = metaworld.MT1(task_name)
            env = mt1.train_classes[task_name]()

            # 2. Get the scripted policy
            policy = self.policy_map[task_name]()

            for ep in tqdm(range(self.ep_per_task)):
                # Sample a random task variation (changes goal/object positions)
                task = np.random.choice(mt1.train_tasks)
                env.set_task(task)

                obs, _ = env.reset()
                done = False
                step = 0

                while not done and step < env.max_path_length:
                    # The scripted policy NEEDS the full observation (including goal)
                    action = policy.get_action(obs)

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # 3. Masking Strategy for FB:
                    # Meta-World v2 observations are 39-D.
                    # [0:36] is the state (hand, object, etc.)
                    # [36:39] is the goal position.
                    # We store them separately to ensure the agent doesn't "see" the goal in s.

                    state_only = obs[:36]
                    next_state_only = next_obs[:36]
                    goal_only = obs[36:39]  # The goal the expert was pursuing

                    self.dataset.append({
                        'obs': state_only.astype(np.float32),
                        'action': action.astype(np.float32),
                        'next_obs': next_state_only.astype(np.float32),
                        'reward': reward,
                        'terminal': terminated,
                        'goal': goal_only.astype(np.float32),
                        'task_id': task_name
                    })

                    obs = next_obs
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
        tasks=['reach-v2', 'push-v2', 'pick-place-v2'],
        episodes_per_task=100
    )
    generator.collect_data()


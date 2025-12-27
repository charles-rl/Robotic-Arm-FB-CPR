import os
import pickle
import numpy as np
import metaworld
from metaworld.policies import (
    SawyerReachV3Policy,
    SawyerPushV3Policy,
    SawyerPickPlaceV3Policy,
    SawyerButtonPressV3Policy
)
from tqdm import tqdm


# --- STEP 1: DATA COLLECTION (Similar to your reference) ---

class FBValidator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.policy_map = {
            'reach-v3': SawyerReachV3Policy,
            'push-v3': SawyerPushV3Policy,
            'pick-place-v3': SawyerPickPlaceV3Policy,
            'button-press-v3': SawyerButtonPressV3Policy
        }

    def get_physics(self, env):
        return {
            'qpos': env.data.qpos.copy(),
            'qvel': env.data.qvel.copy()
        }

    def collect_test_data(self, num_episodes=2):
        dataset = []
        for task_name in self.tasks:
            print(f"Collecting Ground Truth for: {task_name}")
            mt1 = metaworld.MT1(task_name)
            env = mt1.train_classes[task_name]()
            policy = self.policy_map[task_name]()

            for _ in range(num_episodes):
                task = mt1.train_tasks[0]  # Use a fixed variation for testing
                env.set_task(task)
                obs, _ = env.reset()

                for _ in range(20):  # Short snippet
                    action = policy.get_action(obs)
                    physics_before = self.get_physics(env)

                    # The ground truth from the engine
                    next_obs, reward, term, trunc, info = env.step(action)

                    dataset.append({
                        'task_id': task_name,
                        'qpos': physics_before['qpos'],
                        'qvel': physics_before['qvel'],
                        'action': action,
                        'obs': obs,  # 39-D
                        'reward_gt': reward,
                        'goal_gt': obs[36:39]
                    })
                    obs = next_obs
        return dataset

    # --- STEP 2: REWARD INFERENCE (The PhD logic) ---

    def verify_reward_inference(self, dataset):
        print("\nStarting Reward Inference Verification...")
        results = []

        # We create a fresh env to prove we can "teleport" it to a state
        envs = {name: metaworld.MT1(name).train_classes[name]() for name in self.tasks}

        for i, data in enumerate(dataset):
            env = envs[data['task_id']]

            # 1. Teleport Physics
            # We set qpos and qvel to exactly what they were
            # env._reset_hand()
            # env._last_rand_vec = data['goal_gt']
            env._target_pos = data['goal_gt']
            # env.set_task(data['task_id'])
            env.set_state(data['qpos'], data['qvel'])

            # 2. Manually set the environment's internal goal
            # This is the "hidden" part of reward inference.
            # Meta-World reward functions check self._target_pos

            # 3. Synchronize MuJoCo internals
            # This updates the 'data' structure (end-effector pos, etc)
            env.data.site_xpos  # Force a refresh of site positions

            # 4. Get the observation from this teleported state
            # Meta-World v3 uses _get_obs() to construct the 39-D vector
            inferred_obs = env._get_obs()

            # 5. Compute Reward
            # In Meta-World, compute_reward(action, obs) is the logic
            inferred_reward = env.compute_reward(data['action'], inferred_obs)[0]

            # 6. Compare
            diff = abs(inferred_reward - data['reward_gt'])
            results.append(diff)

            if i % 10 == 0:
                print(
                    f"Task: {data['task_id']} | GT: {data['reward_gt']:.4f} | Inf: {inferred_reward:.4f} | Error: {diff:.6f}")

        max_err = max(results)
        print(f"\nVerification Finished.")
        print(f"Max Error across all tasks: {max_err:.8f}")

        if max_err < 1e-5:
            print("SUCCESS: Reward Inference is accurate!")
        else:
            print("FAILURE: Reward discrepancy detected.")


if __name__ == "__main__":
    tasks = ['reach-v3', 'push-v3', 'pick-place-v3', 'button-press-v3']
    tasks = ['reach-v3']
    tester = FBValidator(tasks)

    # Run the validation
    test_data = tester.collect_test_data(num_episodes=2)
    tester.verify_reward_inference(test_data)
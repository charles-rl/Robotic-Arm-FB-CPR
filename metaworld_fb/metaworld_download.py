import pickle
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def extract_lightweight_data(output_file="fb_state_dataset.pkl"):
    print("Initializing stream (no download yet)...")

    # 1. Use streaming=True to avoid downloading the whole repo
    ds = load_dataset("lerobot/metaworld_mt50", split="train", streaming=True)

    # 2. Identify the tasks you want
    # reach: 0, push: 1, pick-place: 2, button-press: 13
    target_tasks = {0, 1, 2, 13}

    # 3. Remove the image column entirely before iteration
    # This ensures the image data is never even requested from the server
    ds = ds.remove_columns(["observation.image"])

    processed_data = []
    max_episodes_per_task = 50  # Adjust this based on how much data you need
    task_counts = {t: 0 for t in target_tasks}

    print("Extracting state-only transitions...")

    for entry in tqdm(ds):
        t_idx = entry["task_index"]

        if t_idx in target_tasks:
            # Check if we have enough for this task (optional limit)
            # if task_counts[t_idx] >= max_episodes_per_task * 500: # approx frames
            #     continue

            # Extract 39-D environment state
            # obs[:36] = State, obs[36:39] = Goal
            env_state = np.array(entry["observation.environment_state"], dtype=np.float32)

            processed_data.append({
                'state': env_state[:36],
                'goal': env_state[36:39],
                'action': np.array(entry["action"], dtype=np.float32),
                'reward': entry["next.reward"],
                'task_idx': t_idx,
                'episode_idx': entry["episode_index"]
            })

            task_counts[t_idx] += 1

            # Exit early if you have enough data for all 4 tasks
            if all(count > 15000 for count in task_counts.values()):
                break

    print(f"Extraction complete. Total transitions: {len(processed_data)}")

    # 4. Save to a compact Pickle file (will likely be < 100MB)
    with open(output_file, "wb") as f:
        pickle.dump(processed_data, f)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    extract_lightweight_data()
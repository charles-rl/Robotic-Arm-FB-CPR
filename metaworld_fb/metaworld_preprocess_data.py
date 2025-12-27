"""
Download code for lerobot dataset is:
hf download lerobot/metaworld_mt50 --repo-type dataset --local-dir ./metaworld_data


--- Scanning Task Indices in Dataset ---
Index 40 matches: push-v3 (found in dataset)
Index 44 matches: reach-wall-v3 (found in dataset)
Index 42 matches: push-back-v3 (found in dataset)
Index  7 matches: button-press-wall-v3 (found in dataset)
Index  4 matches: button-press-topdown-v3 (found in dataset)
Index  5 matches: button-press-topdown-wall-v3 (found in dataset)
Index  6 matches: button-press-v3 (found in dataset)
Index 28 matches: pick-place-wall-v3 (found in dataset)
Index 41 matches: push-wall-v3 (found in dataset)
Index 30 matches: pick-place-v3 (found in dataset)
Index 43 matches: reach-v3 (found in dataset)
Index 38 matches: stick-push-v3 (found in dataset)
Index 10 matches: coffee-push-v3 (found in dataset)

--- FINAL VERIFICATION ---
✅ reach-v3        : Index 43
✅ push-v3         : Index 40
✅ pick-place-v3   : Index 30
✅ button-press-v3 : Index 6

SUCCESS! Use this index list for filtering: [43, 40, 30, 6]
"""

import os
import torch
import numpy as np
from datasets import load_dataset
import glob
from tqdm import tqdm


def extract_state_only_dataset(
        input_dir="./metaworld_data",
        output_path="fb_metaworld_states.pth",
        target_indices=[43, 40, 30, 6]
):
    """
    input_dir: Where you downloaded the HF dataset
    output_path: Where to save the small processed file (.pth)
    target_indices: [reach, push, pick-place, button-press]
    """

    # 1. Locate parquet files
    parquet_pattern = os.path.join(input_dir, "data/**/*.parquet")
    data_files = glob.glob(parquet_pattern, recursive=True)

    if not data_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    print(f"Opening {len(data_files)} files in streaming mode...")

    # 2. Load with specific columns ONLY
    # This is the secret to speed. It ignores the 'observation.image' blobs.
    ds = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        streaming=True
    ).select_columns([
        "observation.environment_state",
        "action",
        "next.reward",
        "task_index",
        "episode_index",
        "next.success"
    ])

    processed_data = {
        "obs": [],
        "goal": [],
        "action": [],
        "reward": [],
        "task_id": [],
        "episode_id": [],
        "success": []
    }

    print(f"Extracting tasks {target_indices} (Images are being skipped)...")

    # We use a filter to only process rows from our 4 tasks
    filtered_ds = ds.filter(lambda x: x["task_index"] in target_indices)

    # 3. Iterate and collect
    # We'll use a count limit if you just want a sample, or remove the break for all
    count = 0
    try:
        for entry in tqdm(filtered_ds):
            # 39-D vector: [0:36] is state, [36:39] is goal
            full_obs = np.array(entry["observation.environment_state"], dtype=np.float32)

            processed_data["obs"].append(full_obs[:36])
            processed_data["goal"].append(full_obs[36:39])
            processed_data["action"].append(np.array(entry["action"], dtype=np.float32))
            processed_data["reward"].append(float(entry["next.reward"]))
            processed_data["task_id"].append(int(entry["task_index"]))
            processed_data["episode_id"].append(int(entry["episode_index"]))
            processed_data["success"].append(bool(entry["next.success"]))

            count += 1
            # Optional: if you want to limit the size for a quick test
            # if count > 50000: break

    except Exception as e:
        print(f"Stream ended or error: {e}")

    # 4. Convert to Tensors and Save
    print("\nConverting to Tensors...")
    for key in processed_data:
        if key == "success":
            processed_data[key] = torch.tensor(processed_data[key], dtype=torch.bool)
        elif key in ["task_id", "episode_id"]:
            processed_data[key] = torch.tensor(processed_data[key], dtype=torch.long)
        else:
            processed_data[key] = torch.tensor(np.array(processed_data[key]), dtype=torch.float32)

    print(f"Saving to {output_path}...")
    torch.save(processed_data, output_path)
    print(f"Finished! Total transitions: {count}")
    print(f"Final file size: {os.path.getsize(output_path) / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    # SPECIFY YOUR PATHS HERE
    IN_DIR = "../../../metaworld_data"
    OUT_FILE = "../../../metaworld_data/parsed_data/fb_metaworld_4tasks_states.pth"
    MY_TASKS = [43, 40, 30, 6]

    extract_state_only_dataset(IN_DIR, OUT_FILE, MY_TASKS)

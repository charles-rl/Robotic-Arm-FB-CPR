import numpy as np
import glob
import os
import argparse
from tqdm import tqdm


def merge_datasets(task_name, chunk_dir, output_dir):
    """
    Reads all npz chunks for a task and merges them into one file.
    """
    # 1. Find all chunks
    search_pattern = os.path.join(chunk_dir, f"{task_name}_chunk_*.npz")
    files = sorted(glob.glob(search_pattern))

    if len(files) == 0:
        print(f"❌ No files found for pattern: {search_pattern}")
        return

    print(f"found {len(files)} chunks for task '{task_name}'. Merging...")

    # 2. Initialize containers
    # We read the first file to know what keys exist (obs, action, physics, etc.)
    first_chunk = np.load(files[0])
    keys = list(first_chunk.keys())
    merged_data = {k: [] for k in keys}
    total_steps = 0

    # 3. Load and Append
    for f in tqdm(files, desc="Loading chunks"):
        with np.load(f) as data:
            for k in keys:
                merged_data[k].append(data[k])
            total_steps += len(data[keys[0]])

    # 4. Concatenate (Flatten the list of arrays into one big array)
    print(f"Concatenating {total_steps} total steps...")
    final_data = {}
    for k in keys:
        final_data[k] = np.concatenate(merged_data[k], axis=0)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{task_name}_merged.npz")

    print(f"💾 Saving to {output_filename}...")
    np.savez_compressed(output_filename, **final_data)
    print("✅ Done!")


if __name__ == "__main__":
    # You can change these defaults or use command line args
    # Example usage: python merge_datasets.py --task lift

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="lift", help="Task name (lift or reach)")
    parser.add_argument("--input_dir", type=str, default="../data/lift", help="Where chunks are saved")
    parser.add_argument("--output_dir", type=str, default="../data", help="Where to save final file")

    args = parser.parse_args()

    # If using the default directory structure from train.py:
    # Input is usually: ../datasets/{task}
    # Output is usually: ../datasets

    if args.input_dir == "../datasets/lift" and args.task != "lift":
        args.input_dir = f"../datasets/{args.task}"

    merge_datasets(args.task, args.input_dir, args.output_dir)

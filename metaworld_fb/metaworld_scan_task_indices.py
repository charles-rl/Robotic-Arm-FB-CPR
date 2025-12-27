import metaworld
from datasets import load_dataset
import glob
import os


def verify_v3_indices(local_path="./metaworld_data"):
    # 1. Get official MT50 task names with v3 suffix
    # The latest Metaworld uses v3 for the Gymnasium-compatible versions
    mt50 = metaworld.MT50()
    all_task_names = list(mt50.train_classes.keys())

    # Let's also check for v3 explicitly if v2 doesn't show up
    all_v3_names = [name.replace('-v2', '-v3') for name in all_task_names]

    # 2. Locate and load local parquet files
    parquet_pattern = os.path.join(local_path, "data/**/*.parquet")
    data_files = glob.glob(parquet_pattern, recursive=True)
    if not data_files:
        print(f"Error: No parquet files found in {local_path}")
        return

    print(f"Found {len(data_files)} parquet files. Loading...")
    ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

    # 3. Targets to find
    target_search = ["reach", "push", "pick-place", "button-press"]
    found_mapping = {}

    print("\n--- Scanning Task Indices in Dataset ---")
    scanned_indices = set()

    # We iterate and match unique task_indices to the library order
    for i, entry in enumerate(ds):
        idx = entry["task_index"]
        if idx not in scanned_indices:
            scanned_indices.add(idx)

            # Identify the name based on the library's internal order
            if idx < len(all_task_names):
                v2_name = all_task_names[idx]
                v3_name = all_v3_names[idx]

                # Check if this matches any of our target keywords
                for target in target_search:
                    if target in v2_name:
                        found_mapping[v2_name] = idx
                        print(f"Index {idx:2d} matches: {v3_name} (found in dataset)")

        # Exit if we found all 50 indices or scanned enough rows
        if len(scanned_indices) >= 50 or i > 100000:
            break

    print("\n--- FINAL VERIFICATION ---")
    # Define exact targets with v3 suffix
    targets = ["reach-v3", "push-v3", "pick-place-v3", "button-press-v3"]
    final_indices = []

    for t in targets:
        v2_t = t.replace("-v3", "-v2")
        # Check both v2 and v3 in our found mapping
        idx = found_mapping.get(v2_t) or found_mapping.get(t)
        if idx is not None:
            print(f"✅ {t:15} : Index {idx}")
            final_indices.append(idx)
        else:
            print(f"❌ {t:15} : NOT FOUND")

    if len(final_indices) == 4:
        print(f"\nSUCCESS! Use this index list for filtering: {final_indices}")
    else:
        print("\nNote: If reach-v3 is still missing, it might be due to the '49 tasks' issue.")
        print("Try looking for 'reach-v2' (index 0) or 'push-puck-v3'.")


if __name__ == "__main__":
    verify_v3_indices("./metaworld_data")

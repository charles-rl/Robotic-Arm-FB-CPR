import wandb
import pandas as pd
import numpy as np

# Replace with your info
ENTITY = "charlessosmena0-academia-sinica"
PROJECT = "so101-lift-fb"
RUN_ID = "vygvg6vw"  # The 8-character ID (e.g., 'ab12cd34')

api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

# Fetching full history as a dataframe is safer
df = run.history()

# Print the available columns so you can see exactly what they are named
print("Available Keys:", df.columns.tolist())

# Select common TQC/Stable-Baselines3 keys (they usually drop the folder name in history)
important_keys = [
    "rollout/ep_rew_mean",
    "eval/mean_reward",
    "train/ent_coef",
    "train/actor_loss",
    "train/loss"
]

for key in important_keys:
    if key in df.columns:
        latest_val = df[key].dropna().iloc[-1]
        mean_val = df[key].dropna().tail(20).mean()
        print(f"{key} -> Latest: {latest_val:.4f}, Recent Mean: {mean_val:.4f}")

print(f"--- Summary for Run: {RUN_ID} ---")

# 1. Standard SB3/TQC Metrics
important_keys = [
    "rollout/ep_rew_mean",
    "eval/mean_reward",
    "train/ent_coef",
    "train/actor_loss",
    "train/critic_loss"
]

for key in important_keys:
    if key in df.columns:
        # Drop NaNs to get real data points
        series = df[key].dropna()
        if not series.empty:
            latest = series.iloc[-1]
            # Compare first 20% vs last 20% to see the trend
            start_mean = series.head(max(1, len(series) // 5)).mean()
            end_mean = series.tail(max(1, len(series) // 5)).mean()
            trend = "RISING" if end_mean > start_mean else "FALLING"

            print(f"{key:20} | Latest: {latest:10.4f} | Trend: {trend}")

# 2. Gradient Analysis (Handling the 'dict' problem)
print("\n--- Gradient Magnitude (Estimated from Histograms) ---")
def get_scalar_from_hist(val):
    """
    Extracts a scalar proxy from WandB histogram dictionaries.
    Prioritizes the 'max' or 'min' to detect exploding gradients.
    """
    if isinstance(val, dict):
        # Handle 'packedBins' structure
        if 'packedBins' in val:
            pb = val['packedBins']
            # We return the absolute max range as a proxy for gradient magnitude
            return max(abs(pb.get('min', 0)), abs(pb.get('max', 0)))
        # Handle 'values'/'bins' structure
        if 'values' in val and 'bins' in val:
            return np.max(np.abs(val['bins']))
    return val if isinstance(val, (int, float)) else np.nan

grad_keys = [k for k in df.columns if "gradients/critic" in k and "weight" in k]

for key in grad_keys[:3]:  # Let's look at the first few layers
    # Apply our extractor to the history of this gradient
    grad_magnitudes = df[key].apply(get_scalar_from_hist).dropna()

    if not grad_magnitudes.empty:
        latest_mag = grad_magnitudes.iloc[-1]
        mean_mag = grad_magnitudes.tail(10).mean()
        # Check for exploding gradients
        growth = latest_mag / (grad_magnitudes.iloc[0] + 1e-8)

        status = "EXPLODING" if growth > 10 else "STABLE"
        print(f"{key:40} | Mag: {latest_mag:.6e} | Status: {status}")


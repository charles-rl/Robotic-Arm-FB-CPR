import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
ENTITY = "charlessosmena0-academia-sinica"  # <--- REPLACE THIS
PROJECT = "so101-lift-fb"  # <--- REPLACE THIS IF DIFFERENT

# Define the runs you want to compare
# ID is the random code in the URL (e.g., 'wandb.ai/user/proj/runs/abc12345' -> 'abc12345')
runs_config = [
    {
        "id": "p79cjxkl",  # REPLACE WITH SAC RUN ID
        "name": "SAC (SB3)",
        "x_key": "env_steps",  # The X-axis you created
        "y_key": "rollout/raw_episode_reward"
    },
    {
        "id": "5rs5tmqt",  # REPLACE WITH TD-MPC RUN ID
        "name": "TD-MPC2",
        "x_key": "_step",  # TD-MPC2 usually uses 'step' or 'episode'
        "y_key": "train/episode_reward"
    }
]


def get_run_data(run_id, x_key, y_key, name):
    api = wandb.Api()
    # Handle the path correctly
    run_path = f"{ENTITY}/{PROJECT}/{run_id}"
    print(f"--> Fetching: {run_path}")

    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"❌ Connection Failed: Could not find run {run_id}. Check ID and Entity/Project names.")
        raise e

    # 1. Download history
    # We add "_step" as a backup X-axis because WandB ALWAYS has this
    history = run.history(keys=[x_key, y_key, "_step"])

    # 2. DEBUGGING: Check what we actually got
    if x_key not in history.columns:
        print(f"⚠️  WARNING for {name}: Could not find X-axis '{x_key}'.")
        print(f"    Available columns are: {list(history.columns)}")
        print(f"    Falling back to internal WandB '_step'.")
        x_key = "_step"  # Fallback

    if y_key not in history.columns:
        print(f"❌ ERROR for {name}: Could not find Y-axis '{y_key}'.")
        print(f"    Available columns are: {list(history.columns)}")
        # We return empty DF to prevent crash, loop will skip it
        return pd.DataFrame()

    # 3. Rename columns for plotting
    # We use a copy to avoid SettingWithCopy warnings
    df = history[[x_key, y_key]].copy()
    df = df.rename(columns={x_key: "Step", y_key: "Reward"})
    df["Algorithm"] = name

    # 4. Clean data
    df = df.dropna()
    print(f"✅ Loaded {name}: {len(df)} rows.")
    return df


def plot_comparison():
    print("Downloading data from WandB...")
    dataframes = []

    for rc in runs_config:
        try:
            df = get_run_data(rc['id'], rc['x_key'], rc['y_key'], rc['name'])
            dataframes.append(df)
            print(f"✅ Loaded {rc['name']} ({len(df)} points)")
        except Exception as e:
            print(f"❌ Could not load {rc['name']}: {e}")

    if not dataframes:
        return

    # Combine all data
    full_df = pd.concat(dataframes)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")

    # Lineplot with automatic smoothing
    sns.lineplot(
        data=full_df,
        x="Step",
        y="Reward",
        hue="Algorithm",
        estimator=None,  # Show raw noisy data
        alpha=0.3  # Make raw lines transparent
    )

    # Add a smoothed trend line (Moving Average)
    # We do this by calculating rolling mean manually for a cleaner plot
    for name in full_df["Algorithm"].unique():
        subset = full_df[full_df["Algorithm"] == name].sort_values("Step")
        # Rolling window of 10 episodes
        subset["Smoothed"] = subset["Reward"].rolling(window=10).mean()
        plt.plot(subset["Step"], subset["Smoothed"], linewidth=2, label=f"{name} (Smoothed)")

    plt.title("SAC vs TD-MPC2: Reach Task", fontsize=16)
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_comparison()

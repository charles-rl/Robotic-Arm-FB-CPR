import numpy as np
import gymnasium
import mujoco
from src.environment_test import SO101LiftEnv


def verify_reward_inference():
    print("=== 1. Starting Data Collection (Online Phase) ===")

    # 1. Initialize Environment
    # We use 'dense' reward because it's more sensitive to physics fluctuations
    env = SO101LiftEnv(render_mode=None, reward_type="dense", control_mode="delta_end_effector")
    env.reset(seed=42)

    recorded_data = []

    # 2. Run an Episode
    done = False
    step_count = 0

    while not done and step_count < 200:
        # Generate a random action (or a heuristic one to actually grasp)
        action = env.action_space.sample()

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Capture the state and result
        # Note: info['physics'] contains qpos + qvel AFTER the step
        step_data = {
            "step": step_count,
            "physics": info["physics"].copy(),  # Vital: Copy to avoid reference issues
            "action": action.copy(),
            "reward": reward
        }
        recorded_data.append(step_data)

        done = terminated or truncated
        step_count += 1

    print(f"Collected {len(recorded_data)} steps of data.")
    print("\n=== 2. Starting Reward Inference (Offline Phase) ===")

    # 3. Verification Loop
    discrepancies = 0

    for i, data in enumerate(recorded_data):
        target_physics = data["physics"]
        target_action = data["action"]
        original_reward = data["reward"]

        # A. Reset Context (Set qpos/qvel directly)
        env.set_physics_state(target_physics)

        # B. Re-calculate Reward
        # We pass the action because the function signature requires it,
        # even if your dense reward relies mostly on state.
        recalculated_reward = env._compute_reward(target_action)

        # C. Compare
        # We use a small epsilon because floating point math can vary slightly
        # depending on CPU optimizations, but they should be virtually identical.
        if not np.isclose(original_reward, recalculated_reward, atol=1e-5):
            print(f"❌ Mismatch at step {i}:")
            print(f"   Original: {original_reward:.6f}")
            print(f"   Recalc:   {recalculated_reward:.6f}")
            print(f"   Diff:     {abs(original_reward - recalculated_reward):.6f}")
            discrepancies += 1
        else:
            # Optional: Print success for first few steps
            if i < 3:
                print(f"✅ Step {i} matched. Reward: {original_reward:.4f}")

    if discrepancies == 0:
        print("\n🎉 SUCCESS: Reward Inference is perfectly deterministic!")
        print("Your environment is compatible with MetaMotivo/FB-style inference.")
    else:
        print(f"\n⚠️ FAILURE: Found {discrepancies} mismatches.")


if __name__ == "__main__":
    verify_reward_inference()
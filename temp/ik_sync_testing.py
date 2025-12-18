import mujoco
import mink
import numpy as np
import time

# Create a simple 2-link arm xml string for testing if you don't want to load your file
# Or just point this to your scene.xml
_XML_PATH = "../simulation/scene.xml"


def test_synchronization():
    print("--- Starting Synchronization Test ---")

    # 1. Setup
    try:
        model = mujoco.MjModel.from_xml_path(_XML_PATH)
    except Exception as e:
        print(f"Could not load your XML at {_XML_PATH}.\nPlease edit the path in the script.")
        return

    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    # 2. Set robot to a pose where gravity will act on it (stretched out)
    # Reset to home or a specific pose
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    else:
        # Manually set a pose if no keyframe (approximate for 6DOF)
        data.qpos[1] = -1.57  # Lift shoulder
        data.qpos[2] = 1.57  # Bend elbow

    mujoco.mj_forward(model, data)

    # 3. SYNC INITIAL STATE
    # At this exact moment, they are equal
    configuration.update(data.qpos)

    print(f"Initial Shoulder Pos (Real): {data.qpos[1]:.4f}")
    print(f"Initial Shoulder Pos (Mink): {configuration.q[1]:.4f}")

    # 4. SIMULATE PHYSICS (The "Step")
    # We step physics forward 50 times. Gravity will pull the arm down.
    # However, we DO NOT touch 'configuration'.
    print("\n... Stepping Physics (Gravity is pulling the arm) ...")
    for _ in range(50):
        # Apply Gravity compensation to 0 so it definitely droops
        data.qfrc_applied[:] = 0
        mujoco.mj_step(model, data)

    # 5. COMPARE
    # data.qpos has changed (gravity). configuration.q has NOT.
    real_pos = data.qpos[1]
    mink_pos = configuration.q[1]
    diff = abs(real_pos - mink_pos)

    print(f"Post-Step Shoulder Pos (Real): {real_pos:.4f}")
    print(f"Post-Step Shoulder Pos (Mink): {mink_pos:.4f}")
    print(f"DIFFERENCE: {diff:.6f}")

    if diff > 1e-4:
        print("\n[CONCLUSION]: Drastic drift detected!")
        print("The Mink internal state thinks the robot is still up high.")
        print("The Real robot has fallen down.")
        print("If you run IK now without .update(), the solver will calculate velocities")
        print("based on the OLD position, causing the robot to 'snap' or explode.")
        print("--> You MUST use self.configuration.update(self.data.qpos)!")
    else:
        print("\n[CONCLUSION]: No drift? (Did the robot not move?)")


if __name__ == "__main__":
    test_synchronization()
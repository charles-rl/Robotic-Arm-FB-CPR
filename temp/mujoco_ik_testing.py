import time
import mujoco
import mujoco.viewer
import numpy as np
import mink
import pygame  # <--- Added Pygame

def apply_rotation(curr_quat, axis, angle):
    """
    Applies a local rotation to the current quaternion.
    axis: list or array [x, y, z] (e.g., [1, 0, 0] for X-axis)
    angle: rotation angle in radians
    """
    # Create the rotation quaternion for this step
    # q = [cos(a/2), sin(a/2)*x, sin(a/2)*y, sin(a/2)*z]
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    rot_quat = np.array([c, s * axis[0], s * axis[1], s * axis[2]])

    # Multiply: new = old * rotation (Local rotation)
    # We use Mujoco's built-in math function for safety
    res = np.zeros(4)
    mujoco.mju_mulQuat(res, curr_quat, rot_quat)
    return res


# ==========================================
# 1. Custom Rate Limiter
# ==========================================
class SimpleRateLimiter:
    def __init__(self, frequency):
        self.period = 1.0 / frequency
        self.last_time = time.perf_counter()

    def sleep(self):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_time
        remaining = self.period - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_time = time.perf_counter()

    @property
    def dt(self):
        return self.period


# ==========================================
# 2. Setup
# ==========================================
_XML = "../simulation/scene.xml"

if __name__ == "__main__":
    # --- Pygame Setup for Input ---
    pygame.init()
    # Create a small 300x300 window to capture keystrokes
    screen = pygame.display.set_mode((300, 300))
    pygame.display.set_caption("Control Input (Focus Here)")

    # Text setup for the pygame window
    font = pygame.font.SysFont("Arial", 18)

    # --- Mujoco Setup ---
    model = mujoco.MjModel.from_xml_path(_XML)
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    # --- Tasks ---
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="gripperframe",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.05,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model, cost=1e-2),
    ]

    # --- Limits ---
    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow_flex": np.pi,
        "wrist_flex": np.pi,
        "wrist_roll": np.pi,
        "gripper": np.pi,
    }

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
        # mink.VelocityLimit(model, max_velocities),
    ]

    # --- IK Settings ---
    target_mocap_name = "target"
    mid = model.body(target_mocap_name).mocapid[0]

    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20  # 10 in simulation for rl
    move_arm = -1.0

    # Launch Passive Viewer (No key_callback needed anymore)
    with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        # Move mocap to current effector pos
        mink.move_mocap_to_frame(model, data, target_mocap_name, "gripperframe", "site")

        rate = SimpleRateLimiter(frequency=60.0)

        # Movement speed (meters per tick)
        move_speed = 0.005

        running = True
        while viewer.is_running() and running:

            # ==========================================
            # 3. Pygame Input Handling (Continuous)
            # ==========================================
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get the state of all keys
            keys = pygame.key.get_pressed()
            mocap_offset = np.zeros(3)

            # Map Keys to XYZ
            # X Axis
            if keys[pygame.K_w]:
                mocap_offset[0] += move_speed
            if keys[pygame.K_s]:
                mocap_offset[0] -= move_speed

            # Y Axis
            if keys[pygame.K_a]:  # Left/Right might vary based on your camera view
                mocap_offset[1] += move_speed
            if keys[pygame.K_d]:
                mocap_offset[1] -= move_speed

            # Z Axis
            if keys[pygame.K_SPACE]:
                mocap_offset[2] += move_speed
            if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                mocap_offset[2] -= move_speed

            # Apply offset to Mocap
            if np.any(mocap_offset):
                data.mocap_pos[mid] += mocap_offset

            # --- Orientation Control (Arrows + Q/E) ---
            rot_speed = 0.02  # Radians per tick

            # Roll (X-axis) - Q / E
            if keys[pygame.K_q]:
                data.mocap_quat[mid] = apply_rotation(data.mocap_quat[mid], [1, 0, 0], -rot_speed)
            if keys[pygame.K_e]:
                data.mocap_quat[mid] = apply_rotation(data.mocap_quat[mid], [1, 0, 0], rot_speed)

            # Pitch (Y-axis) - Up / Down
            if keys[pygame.K_UP]:
                data.mocap_quat[mid] = apply_rotation(data.mocap_quat[mid], [0, 1, 0], rot_speed)
            if keys[pygame.K_DOWN]:
                data.mocap_quat[mid] = apply_rotation(data.mocap_quat[mid], [0, 1, 0], -rot_speed)

            # Yaw (Z-axis) - Left / Right
            if keys[pygame.K_LEFT]:
                data.mocap_quat[mid] = apply_rotation(data.mocap_quat[mid], [0, 0, 1], rot_speed)
            if keys[pygame.K_RIGHT]:
                data.mocap_quat[mid] = apply_rotation(data.mocap_quat[mid], [0, 0, 1], -rot_speed)

            # ==========================================
            # 4. Mink IK Pipeline
            # ==========================================
            T_wt = mink.SE3.from_mocap_name(model, data, target_mocap_name)
            end_effector_task.set_target(T_wt)

            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)

                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            # q_index and the i is just the same location
            if keys[pygame.K_1]:
                move_arm = 1.0
            if keys[pygame.K_2]:
                move_arm = -1.0
            data.ctrl[:5] = configuration.q[:5]
            data.ctrl[-1] = move_arm

            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()

    pygame.quit()
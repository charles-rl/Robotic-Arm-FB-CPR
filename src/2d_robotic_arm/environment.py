import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np

def simp_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def simulation2theoretical_angles(angles):
    """
    :param angles: [l1, l2, l3]: numpy array, angles are relative to the previous link's angles
    :return:
    """
    if isinstance(angles, list):
        angles = np.array(angles)

    angle1 = -angles[0]
    angle2 = -angles[0] + angles[1]
    angle3 = -angles[2] + np.pi + angles[1]
    return np.array([angle1, angle2, angle3])

def theoretical2simulation_angles(angles):
    """
    :param angles: [l1, l2, l3]: numpy array, angles are relative to the global positive x-axis direction
    :return:
    """
    if isinstance(angles, list):
        angles = np.array(angles)

    angle1 = -angles[0]
    angle2 = -angles[0] - angles[1]
    angle3 = -angles[0] - angles[1] - angles[2] + np.pi
    return np.array([angle1, angle2, angle3])

def forward_kinematics(t1, t2, t3, l1, l2, l3):
    x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2) + l3 * np.cos(t1 + t2 + t3)
    y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2) + l3 * np.sin(t1 + t2 + t3)
    return np.array([x, y])

def inverse_kinematics(target, lengths, curr_angles, min_error=1e-3, max_iterations=10000, alpha=0.05):
    t1, t2, t3 = curr_angles
    l1, l2, l3 = lengths
    for _ in range(max_iterations):
        current_pos = forward_kinematics(t1, t2, t3, l1, l2, l3)
        error = target - current_pos
        if np.linalg.norm(error) < min_error:
            break

        # Jacobian transpose method
        s1 = np.sin(t1)
        c1 = np.cos(t1)
        s12 = np.sin(t1 + t2)
        c12 = np.cos(t1 + t2)
        s123 = np.sin(t1 + t2 + t3)
        c123 = np.cos(t1 + t2 + t3)
        J_t = np.array([
            [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
            [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123]
        ]).T
        delta_theta = alpha * J_t @ error
        t1 += delta_theta[0]
        t2 += delta_theta[1]
        t3 += delta_theta[2]
    return np.array([t1, t2, t3])


class PIDController:
    def __init__(self, kp, kd, ki, dt):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt
        self.i_err = 0.0

    def get_value(self, error, velocity):
        # It is supposed to be
        # val = kp * err - kd * vel + ki * i_err
        # but it is opposite in this env
        # val = -kp * err + kd * vel - ki * i_err
        self.i_err += error * self.dt
        # print(-self.kp * error, self.kd * velocity, -self.i_err)
        val = -self.kp * error + self.kd * velocity - self.ki * self.i_err
        if abs(val) < 0.01:
            val = 0.0
            self.i_err = 0.0
        return val

    def position_to_angles(self, x, y, lengths, curr_angles):
        # Simulation to theoretical
        target_angles = inverse_kinematics(np.array([x, y]),
                                           lengths,
                                           simulation2theoretical_angles(curr_angles))
        return target_angles

    def do_angles(self, desired_angles, angles, angular_velocities):
        """
        must be in order arm1, arm2, arm3
        base to tip of arm
        must be numpy
        :param desired_angles:
        :param angles:
        :param angular_velocities:
        :return:
        """
        error1 = desired_angles[0] - angles[0]
        velo1 = angular_velocities[0]
        value1 = self.get_value(error1, velo1)

        error2 = desired_angles[1] - angles[1]
        velo2 = angular_velocities[1]
        value2 = self.get_value(error2, velo2)

        error3 = desired_angles[2] - angles[2]
        velo3 = angular_velocities[2]
        value3 = self.get_value(error3, velo3)

        # print(abs(error1), abs(error2), abs(error3), abs(velo1), abs(velo2), abs(velo3))
        errors = np.array([error1, error2, error3])
        velocities = np.array([velo1, velo2, velo3])
        min_errors = np.array([0.05, 0.05, 0.15])
        min_velos = np.array([0.1, 0.1, 0.1])
        is_stop_motors = bool(np.logical_and(np.all(abs(errors) < min_errors), np.all(abs(velocities) < min_velos)))
        return np.array([value1, value2, value3]), is_stop_motors


class RobotArmEnv:
    MAX_TIMESTEPS = 1000
    FPS = 60
    WIDTH, HEIGHT = 800, 600
    # Everything will be relative to the robotic arm's base, that is the 'origin' point
    ARM_LENGTH = 140
    ARM_RADIUS = 25
    MAX_FORCE = 25_000_000
    # Table base position
    TABLE_RADIUS = 10
    BASE_X, BASE_Y = WIDTH // 2, HEIGHT - TABLE_RADIUS // 2

    # Color scheme - minimalistic dark theme
    BG_COLOR = (20, 20, 25)  # Very dark blue-gray
    TABLE_COLOR = (60, 60, 70)  # Subtle gray for table
    ARM1_COLOR = (100, 150, 255, 255)  # Soft blue
    ARM2_COLOR = (150, 100, 255, 255)  # Soft purple
    ARM3_COLOR = (255, 100, 150, 255)  # Soft pink
    JOINT_COLOR = (200, 200, 210)  # Light gray for joints
    TEXT_COLOR = (180, 180, 190)  # Muted text
    TARGET_COLOR = (255, 200, 40)

    def __init__(self, render):
        # Initialize Pygame
        pygame.init()
        self.is_render = render
        if render:
            pygame.display.set_caption("2D Robotic Arm - Pymunk")
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.dt = 1 / float(self.FPS)

        # Initialize Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, 9.81)

        # Environment parameters
        # arm 1 reference angle is at the positive x-axis, counter-clockwise negative
        # arm 2 reference angle is at the positive x-axis, counter-clockwise negative
        # arm 3 reference angle is at the negative x-axis, counter-clockwise negative
        self.init_angles = np.array([-np.pi - 0.22, -0.22, -np.pi + 0.22])  # similar to lerobot position
        # self.init_angles = np.array([-np.pi / 2 + 0.1, 0.2, -np.pi // 2 + 0.22])
        # self.init_angles = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2])
        self.target_pos_world = np.array([0, 0])
        self.timesteps = 0
        self.links = []
        self.joints = []
        self.motors = []
        self.observation_space = (11,)
        self.action_space = (3,)
        self.controller = PIDController(2.2, 4.75, 0.01, self.dt)

    def _make_segment(self, position: pymunk.Vec2d, vector: pymunk.Vec2d, radius: int, color: tuple):
        body = pymunk.Body()
        body.position = position
        shape = pymunk.Segment(body, (0, 0), vector, radius)
        shape.density = 0.2
        shape.elasticity = 0.03
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.color = color
        self.space.add(body, shape)
        self.links.append(body)

    def _make_pivot_joint(self, body1, body2, point_position1=(0, 0), point_position2=(0, 0), collide=True):
        joint = pymunk.PinJoint(body1, body2, point_position1, point_position2)
        joint.collide_bodies = collide
        self.space.add(joint)
        self.joints.append(joint)

    def _make_simple_motor(self, body1, body2, angular_velocity):
        motor = pymunk.SimpleMotor(body1, body2, angular_velocity)
        motor.max_force = self.MAX_FORCE
        self.space.add(motor)
        self.motors.append(motor)

    def _limit_joint(self, body1, body2, minimum, maximum, collide=True):
        """Minimum and maximum are in rads"""
        joint = pymunk.RotaryLimitJoint(body1, body2, minimum, maximum)
        joint.collide_bodies = collide
        self.space.add(joint)
        # self.joints.append(joint)

    def _setup_env(self):
        # Create static table
        table_body = self.space.static_body
        table_shape = pymunk.Segment(table_body, (0, self.BASE_Y), (self.WIDTH, self.BASE_Y), self.TABLE_RADIUS)
        table_shape.color = pygame.Color(*self.TABLE_COLOR)
        table_shape.friction = 1.0
        self.space.add(table_shape)

        # Create base of arm, a bit higher than the table similar to LeRobot
        elevation_offset = self.ARM_LENGTH  # from LeRobot image (estimate)
        arm1_position, arm1_vector = pymunk.Vec2d(self.BASE_X, self.BASE_Y - elevation_offset), pymunk.Vec2d(self.ARM_LENGTH, 0)
        arm1_body = self.space.static_body
        # order is important, segments must be created before joints
        self._make_segment(arm1_position, arm1_vector, radius=self.ARM_RADIUS + 2, color=self.ARM1_COLOR)
        self._make_pivot_joint(arm1_body, self.links[-1], arm1_position)
        self._make_simple_motor(arm1_body, self.links[-1], 0.0)
        # self._limit_joint(arm1_body, self.links[-1], -1.0, 1.0)

        # Create second arm
        arm2_vector = pymunk.Vec2d(self.ARM_LENGTH, 0)
        self._make_segment(arm1_position + arm2_vector, arm2_vector, radius=self.ARM_RADIUS, color=self.ARM2_COLOR)
        self._make_pivot_joint(self.links[-2], self.links[-1], arm2_vector, (0, 0))
        self._make_simple_motor(self.links[-2], self.links[-1], 0.0)
        # self._limit_joint(self.links[-2], self.links[-1], -1.0, 1.0)

        # Create third arm
        third_arm_length = self.ARM_LENGTH // 2  # from LeRobot image (estimate)
        arm3_vector = pymunk.Vec2d(third_arm_length, 0)
        self._make_segment(arm1_position + arm2_vector + arm3_vector, arm3_vector, radius=self.ARM_RADIUS - 2, color=self.ARM3_COLOR)
        self._make_pivot_joint(self.links[-2], self.links[-1], 2 * arm3_vector, arm3_vector)
        self._make_simple_motor(self.links[-2], self.links[-1], 0.0)
        # self._limit_joint(self.links[-2], self.links[-1], -1.0, 1.0)

    def _move_to_start_position(self, render):
        frames = []
        counter = 0
        while True:
            self.space.step(self.dt)
            # if render:
            #     self.render()
            values, is_stop_motors = self.controller.do_angles(self.init_angles,
                                                               [l.angle for l in self.links],
                                                               [l.angular_velocity for l in self.links])
            for i, motor in enumerate(self.motors):
                motor.rate = values[i]

            if is_stop_motors:
                for motor in self.motors:
                    motor.rate = 0.0
                break

        #     if counter % 2 == 0:
        #         frame_surface = self.screen.copy()
        #         frames.append(np.array(pygame.surfarray.pixels3d(frame_surface)).transpose(1, 0, 2))
        #     counter += 1
        #
        # import imageio
        #
        # print("Saving GIF...")
        # imageio.mimsave("robot_arm_demo.gif", frames, fps=self.FPS // 2)
        # print("GIF saved!")

    def _get_obs(self):
        """
        :return: numpy array and dict of raw angles
        Observation is arranged like this:
        end_pos[0], end_pos[1],
        cos(theta1), sin(theta1), cos(theta2), sin(theta2), cos(theta3), sin(theta3),
        angular velocities for each joint; theta1, theta2, theta3
        """
        n_links = len(self.links)
        angles = np.zeros((n_links * 2,), dtype=np.float32)
        angular_velocities = np.zeros((n_links,), dtype=np.float32)
        arm_raw_angles = np.zeros((n_links,), dtype=np.float32)
        for i, link in enumerate(self.links):
            angles[2 * i] = np.cos(link.angle)
            angles[2 * i + 1] = np.sin(link.angle)
            angular_velocities[i] = link.angular_velocity
            arm_raw_angles[i] = link.angle

        # End effector position
        end_pos = self.links[2].position + pymunk.Vec2d(0, 0).rotated(self.links[2].angle)
        # where the base arm point is located
        # then normalize
        end_pos = (end_pos - pymunk.Vec2d(self.BASE_X, self.BASE_Y - self.ARM_LENGTH)) / self.ARM_LENGTH
        end_pos = np.array([end_pos.x, -end_pos.y], dtype=np.float32)
        obs = np.concatenate([end_pos, angles, angular_velocities])

        motor_rates = np.zeros((n_links,), dtype=np.float32)
        motor_forces = np.zeros((n_links,), dtype=np.float32)
        for i, motor in enumerate(self.motors):
            motor_rates[i] = motor.rate
            motor_forces[i] = motor.impulse / self.dt  # accrd to documentation divide impulse by dt to get force

        info = {"arm_raw_angles": arm_raw_angles, "arm_raw_velocities": angular_velocities, "motor_rates": motor_rates, "motor_forces": motor_forces}
        return obs, info

    def reset(self, render=False):
        # Decided to destroy object instead of resetting
        for body in self.space.bodies:
            self.space.remove(body)
        for shape in self.space.shapes:
            self.space.remove(shape)
        for constraint in self.space.constraints:
            self.space.remove(constraint)

        self.links = []
        self.joints = []
        self.motors = []

        self._setup_env()
        self._move_to_start_position(render)

        return self._get_obs()

    def step(self, action, task_name: str = None):
        """
        :param action: list or numpy between -1 and 1
        # TODO: maybe move to init of env
        :param task_name: change reward based on the task
        :return:
        """
        # Process the event queue to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # TODO: Rescale action and also get the action data again
        if isinstance(action, list):
            action = np.array(action)
        # action = np.clip(action, -1.0, 1.0)
        action *= 20.0

        for i, motor in enumerate(self.motors):
            motor.rate = action[i]
        self.space.step(self.dt)
        self.timesteps += 1

        # TODO: fix depending on task change reward

        reward = self._get_reward(task_name)

        terminated = False
        truncated = bool(self.timesteps >= self.MAX_TIMESTEPS)
        obs, info = self._get_obs()

        return obs, reward, terminated, truncated, info

    def _get_reward(self, task_name):
        # End effector position
        end_pos = self.links[2].position + pymunk.Vec2d(0, 0).rotated(self.links[2].angle)
        # where the base arm point is located
        # then normalize
        end_pos = (end_pos - pymunk.Vec2d(self.BASE_X, self.BASE_Y - self.ARM_LENGTH)) / self.ARM_LENGTH
        end_pos = np.array([end_pos.x, -end_pos.y], dtype=np.float32)

        reward = 0.0
        target_pos = np.array([0.0, 0.0])
        # "reach_center", "reach_top", "reach_right_top", "reach_left_top", "reach_bottom"
        if task_name == "reach_center":
            target_pos = np.array([0.0, 0.0], dtype=np.float32)
        elif task_name == "reach_top":
            target_pos = np.array([0.0, 2.2], dtype=np.float32)
        elif task_name == "reach_right_top":
            target_pos = np.array([1.5, 1.8], dtype=np.float32)
        elif task_name == "reach_left_top":
            target_pos = np.array([-1.5, 1.8], dtype=np.float32)
        elif task_name == "reach_bottom":
            target_pos = np.array([0.0, -0.5], dtype=np.float32)

        if self.is_render:
            target_pos_world_x = int(target_pos[0] * self.ARM_LENGTH) + self.BASE_X
            target_pos_world_y = int(-target_pos[1] * self.ARM_LENGTH) + self.BASE_Y - self.ARM_LENGTH
            self.target_pos_world = np.array([target_pos_world_x, target_pos_world_y])
        reward = -np.linalg.norm(target_pos - end_pos)

        reward /= 2.5
        return reward

    def sample_action(self):
        return np.random.uniform(low=-1.0, high=1.0, size=(3,))

    def _draw_joint_markers(self):
        """Draw small circles at joint positions for visual clarity"""
        joint_radius = 8
        # Base joint
        base_pos = (self.BASE_X, self.BASE_Y - self.ARM_LENGTH)
        pygame.draw.circle(self.screen, self.JOINT_COLOR,
                           (int(base_pos[0]), int(base_pos[1])), joint_radius)

        # Joint between arm 1 and 2
        if len(self.links) >= 2:
            joint_pos = self.links[0].position + pymunk.Vec2d(self.ARM_LENGTH, 0).rotated(self.links[0].angle)
            pygame.draw.circle(self.screen, self.JOINT_COLOR,
                               (int(joint_pos.x), int(joint_pos.y)), joint_radius)

        # Joint between arm 2 and 3
        if len(self.links) >= 3:
            joint_pos = self.links[1].position + pymunk.Vec2d(self.ARM_LENGTH, 0).rotated(self.links[1].angle)
            pygame.draw.circle(self.screen, self.JOINT_COLOR,
                               (int(joint_pos.x), int(joint_pos.y)), joint_radius)

        # End effector (tip of arm 3)
        if len(self.links) >= 3:
            end_pos = self.links[2].position + pymunk.Vec2d(0, 0).rotated(self.links[2].angle)
            pygame.draw.circle(self.screen, (255, 255, 255),
                               (int(end_pos.x), int(end_pos.y)), 6)

    def render(self, render_fps=True):
        """Draws the current state of the environment on the screen."""
        self.screen.fill(pygame.Color(*self.BG_COLOR))
        self.space.debug_draw(self.draw_options)
        self._draw_joint_markers()

        pygame.draw.circle(self.screen, self.TARGET_COLOR,
                           (int(self.target_pos_world[0]), int(self.target_pos_world[1])), 5)

        # Display fps
        # TODO: Set to False if using pixels as input
        if render_fps:
            font = pygame.font.Font(None, 24)
            info_text = font.render(f"FPS: {int(self.clock.get_fps())}", True, self.TEXT_COLOR)
            self.screen.blit(info_text, (10, 10))

        self.clock.tick(self.FPS)

        pygame.display.flip()

    def close(self):
        """Cleans up the Pygame window."""
        pygame.quit()


if __name__ == "__main__":
    env = RobotArmEnv(True)

    done = False
    observation, info = env.reset(render=False)

    # Follow mouse
    target_pos_world = np.array([400, 300])
    target_pos_relative = (target_pos_world - np.array([env.BASE_X, env.BASE_Y - env.ARM_LENGTH])) / env.ARM_LENGTH
    target_pos_relative[1] = -target_pos_relative[1]

    target_angles_theoretical = env.controller.position_to_angles(
        target_pos_relative[0],
        target_pos_relative[1],
        [1.0, 1.0, 0.5],
        info["arm_raw_angles"]
    )
    target_angles = theoretical2simulation_angles(target_angles_theoretical)

    print(target_pos_relative)
    print(info["arm_raw_angles"])
    print(target_angles)
    target_angles_ = []
    for curr_angle, target_angle in zip(info["arm_raw_angles"], target_angles):
        if abs(target_angle - curr_angle) > np.pi:
            target_angles_.append(simp_angle(target_angle))
        else:
            target_angles_.append(target_angle)
    target_angles = np.array(target_angles_)
    print(target_angles)
    env.target_pos_world = target_pos_world

    while not done:
        # action = env.sample_action()
        action = np.array([0.0,0.0,0.0])

        values, is_stop_motors = env.controller.do_angles(target_angles, info["arm_raw_angles"], info["arm_raw_velocities"])

        if not is_stop_motors:
            action = values

        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env.render()
    env.close()

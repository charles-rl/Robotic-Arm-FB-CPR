import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Robotic Arm - Pymunk")
clock = pygame.time.Clock()

# Initialize Pymunk space
space = pymunk.Space()
space.gravity = (0, 0)  # Gravity pointing down

# Drawing options
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Base position (where arm is attached to table)
base_x, base_y = WIDTH // 2, HEIGHT - 50  # middle of screen TODO: maybe edge of screen

# Create static table
ground_body = space.static_body
ground_shape = pymunk.Segment(ground_body, (0, base_y), (WIDTH, base_y), 10)
ground_shape.friction = 1.0
space.add(ground_shape)

# Arm parameters
ARM_LENGTH = 100
ARM_WIDTH = 20
ARM_MASS = 1

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        space.add(joint)

class Segment:
    def __init__(self, p0, v, radius=10):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)

class DampedRotarySpring:
    def __init__(self, b, b2, angle, stiffness, damping):
        joint = pymunk.DampedRotarySpring(
            b, b2, angle, stiffness, damping)
        space.add(joint)


class RotaryLimitJoint:
    def __init__(self, b, b2, min, max, collide=True):
        joint = pymunk.RotaryLimitJoint(b, b2, min, max)
        joint.collide_bodies = collide
        space.add(joint)

class SimpleMotor:
    def __init__(self, b, b2, rate):
        joint = pymunk.SimpleMotor(b, b2, rate)
        space.add(joint)

p0, v = pymunk.Vec2d(300, 350), pymunk.Vec2d(50, 0)
b0 = space.static_body
arm = Segment(p0, v)
arm.body.angular_velocity
arm_joint = PivotJoint(b0, arm.body, p0)
motor = SimpleMotor(b0, arm.body, 0)


arm2 = Segment(p0+v, v)
PivotJoint(arm.body, arm2.body, v, (0, 0))
DampedRotarySpring(arm.body, arm2.body, 0, 10000000, 10000)
RotaryLimitJoint(arm.body, arm2.body, -1, 1)

# Main loop
counter = 0
running = True
dt = 1 / 60.0

frames = []
frame_counter = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False


    if counter > 60 * 2:
        arm.body.angle = 3.14
        counter = 0

    # Step the physics simulation
    space.step(dt)

    # Clear screen
    screen.fill((255, 255, 255))

    # Draw everything
    space.debug_draw(draw_options)

    # Display info
    font = pygame.font.Font(None, 24)
    info_text = font.render(f"FPS: {int(clock.get_fps())}", True, (0, 0, 0))
    screen.blit(info_text, (10, 10))

    if frame_counter % 2 == 0:
        frame_surface = screen.copy()
        frames.append(np.array(pygame.surfarray.pixels3d(frame_surface)).transpose(1, 0, 2))
    frame_counter += 1

    # Update display
    pygame.display.flip()
    clock.tick(60)
    counter += 1

import imageio

print("Saving GIF...")
imageio.mimsave("robot_arm_demo.gif", frames, fps=self.FPS // 2)
print("GIF saved!")

pygame.quit()
sys.exit()



import os
os.environ["WEBOTS_HOME"] = "Z:\Webots"
from controller import Supervisor
import tensorflow as tf
import tensorflow_probability as tfp


class RobotController:
    def __init__(self, simulation_delay, robot_base_speed, screen_divider):
        self.supervisor = Supervisor()
        self.simulation_delay = simulation_delay
        self.base_speed = robot_base_speed
        self.screen_divider = screen_divider

        self.front_left_motor = self.supervisor.getDevice("fl_wheel_joint")
        self.front_right_motor = self.supervisor.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.supervisor.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.supervisor.getDevice("rr_wheel_joint")

        self.camera_depth = self.supervisor.getDevice('camera depth')
        self.camera_depth.enable(50)

        self.camera_observation = None
        self.do_forward = []

    # ////////////////////// ENVIROMENT //////////////////////#
    def setup_motors(self):
        # Set position of motors to infinity
        self.front_left_motor.setPosition(float('inf'))
        self.front_right_motor.setPosition(float('inf'))
        self.rear_left_motor.setPosition(float('inf'))
        self.rear_right_motor.setPosition(float('inf'))
        # Set velocity of motors
        self.front_left_motor.setVelocity(self.base_speed)
        self.front_right_motor.setVelocity(self.base_speed)
        self.rear_left_motor.setVelocity(self.base_speed)
        self.rear_right_motor.setVelocity(self.base_speed)

    def reset(self):
        self.supervisor.simulationReset()
        self.supervisor.step(50)
        self.setup_motors()

        observation = self.get_center_row()
        return observation

    def step(self, action):
        done = False
        if action == 0:
            self.move_forward()
            self.supervisor.step(self.simulation_delay)
            self.do_forward.append(True)
            reward = 1
        elif action == 1:
            self.move_right()
            self.supervisor.step(self.simulation_delay)
            self.do_forward = []
            reward = -0.2
        elif action == 2:
            self.move_left()
            self.supervisor.step(self.simulation_delay)
            self.do_forward = []
            reward = -0.2
        else:
            reward = 0

        observation = self.get_center_row()
        self.observation = observation

        min_val = min(observation)
        if 0 <= min_val <= 0.21:
            reward -= 200
            done = True
        if 0.21 < min_val < 0.4:
            reward -= 0.3
        elif len(self.do_forward) == 2:
            reward += 5

        return observation, reward, done

    # ////////////////////// ENVIROMENT //////////////////////#

    # ////////////////////// ACTIONS //////////////////////#
    def move_forward(self):
        self.front_left_motor.setVelocity(self.base_speed)
        self.front_right_motor.setVelocity(self.base_speed)
        self.rear_left_motor.setVelocity(self.base_speed)
        self.rear_right_motor.setVelocity(self.base_speed)

    def move_left(self):
        self.front_left_motor.setVelocity(self.base_speed*0.1)
        self.front_right_motor.setVelocity(self.base_speed)
        self.rear_left_motor.setVelocity(self.base_speed*0.1)
        self.rear_right_motor.setVelocity(self.base_speed)

    def move_right(self):
        self.front_left_motor.setVelocity(self.base_speed)
        self.front_right_motor.setVelocity(self.base_speed*0.1)
        self.rear_left_motor.setVelocity(self.base_speed)
        self.rear_right_motor.setVelocity(self.base_speed*0.1)

    # ////////////////////// ACTIONS //////////////////////#

    # ///////////////////////////// CAMERA ////////////////////////#
    def get_center_row(self):
        depth_image = self.camera_depth.getRangeImage()
        center_row = depth_image[480 // 2 * 640:(480 // 2 + 1) * 640:self.screen_divider]
        self.camera_observation = center_row
        return center_row
    # ///////////////////////////// CAMERA ////////////////////////#


if __name__ == "__main__":
    control = RobotController()
    control.get_center_row()

    control.setup_motors()


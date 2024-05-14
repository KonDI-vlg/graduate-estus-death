import os

os.environ["WEBOTS_HOME"] = "/usr/local/webots"
from controller import Supervisor


class RobotController:
    def __init__(self):
        self.supervisor = Supervisor()
        self.base_speed = 1

        self.front_left_motor = self.supervisor.getDevice("fl_wheel_joint")
        self.front_right_motor = self.supervisor.getDevice("fr_wheel_joint")
        self.rear_left_motor = self.supervisor.getDevice("rl_wheel_joint")
        self.rear_right_motor = self.supervisor.getDevice("rr_wheel_joint")

        self.camera_depth = self.supervisor.getDevice('camera depth')
        self.camera_depth.enable(1500)

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

    def get_action(self):
        data = self.get_center_row()

    def reset_environment(self):
        self.supervisor.simulationReset()
        self.supervisor.step(50)
        self.setup_motors()

    def move_forward(self):
        self.front_left_motor.setVelocity(self.base_speed)
        self.front_right_motor.setVelocity(self.base_speed)
        self.rear_left_motor.setVelocity(self.base_speed)
        self.rear_right_motor.setVelocity(self.base_speed)


    def turn_left(self):
        self.front_left_motor.setVelocity(-self.base_speed)
        self.front_right_motor.setVelocity(self.base_speed)
        self.rear_left_motor.setVelocity(-self.base_speed)
        self.rear_right_motor.setVelocity(self.base_speed)

    def turn_right(self):
        self.front_left_motor.setVelocity(self.base_speed)
        self.front_right_motor.setVelocity(-self.base_speed)
        self.rear_left_motor.setVelocity(self.base_speed)
        self.rear_right_motor.setVelocity(-self.base_speed)

    # ///////////////////////////// CAMERA ////////////////////////#
    def get_center_row(self):
        if self.supervisor.step(1500) != -1:
            width = self.camera_depth.getWidth()
            height = self.camera_depth.getHeight()
            depth_image = self.camera_depth.getRangeImage()
            center_row = depth_image[height // 2 * width:(height // 2 + 1) * width]
            return center_row
        return None


# Example of usage
if __name__ == "__main__":
    control = RobotController()
    control.reset_environment()
    control.setup_motors()
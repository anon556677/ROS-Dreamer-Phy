#!/usr/bin/env python

import rospy
import numpy as np
from ackermann_msgs.msg import AckermannDrive
from dreamer2ros.msg import Act2

class Smooth:
    def __init__(self):
        self.cmd_pub = rospy.Publisher('cmd_drive', AckermannDrive, queue_size=1)
        rospy.Subscriber('cmd_rl', Act2, self.cmdCallback)
        self.rate = rospy.get_param('~rate',50)
        self.beta = rospy.get_param('~beta',0.2)
        self.speed_target = 0.
        self.steering_target = 0.
        self.cmd = AckermannDrive()

    def cmdCallback(self, data):
        self.steering_target = self.check_and_saturate(data.a0, self.steering_target)
        self.speed_target = self.check_and_saturate(data.a1, self.speed_target)

    def check_and_saturate(self, value, target):
        if np.isnan(value):
            target = 0
        elif value > 1.0:
            target = 1.0
        elif value < -1.0:
            target = -1.0
        else:
            target = value
        return target

    def update(self, current, target):
        if abs(current - target) > self.beta:
            current = current + np.sign(- current + target)*self.beta
        else:
            current = target
        return current

    def run(self):
        rate = rospy.Rate(self.rate)
        self.cmd.steering_angle = 0.0
        self.cmd.speed = 0.0
        while not rospy.is_shutdown():
            self.cmd.steering_angle = self.update(self.cmd.steering_angle, self.steering_target)
            self.cmd.speed = self.update(self.cmd.speed, self.speed_target)
            self.cmd_pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('smoother')
    S = Smooth()
    S.run()

#!/usr/bin/env python

import rospy
import numpy as np
from dreamer2ros.msg import Act2
from geometry_msgs.msg import Drive

class Smooth:
    def __init__(self):
        self.cmd_pub = rospy.Publisher('cmd_drive', Drive, queue_size=1)
        rospy.Subscriber('cmd_rl', Act2, self.cmdCallback)
        self.rate = rospy.get_param('~rate',50)
        self.beta = rospy.get_param('~beta',0.2)
        self.wheel_offset = rospy.get_param('~wheel_offset') # offset for each wheel from front axis in meters
        self.left_target = 0.
        self.right_target = 0.
        self.cmd = Drive()

    def check_and_saturate(self, value):
        if np.isnan(value):
            target = 0
        elif value > 1.0:
            target = 1.0
        elif value < -1.0:
            target = -1.0
        else:
            target = value
        return target

    def map_cmd(self, linear, angular):
        left  = linear + self.wheel_offset * angular
        right = linear - self.wheel_offset * angular
        return left, right
        
    def cmdCallback(self, data):
        linear = self.check_and_saturate(data.a0)
        angular = self.check_and_saturate(data.a1)
        self.left_target, self.right_target = self.map_cmd(linear, angular)

    def update(self, current, target):
        if abs(current - target) > self.beta:
            current = current + np.sign(- current + target)*self.beta
        else:
            current = target
        return current

    def run(self):
        rate = rospy.Rate(self.rate)
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        while not rospy.is_shutdown():
            self.cmd.left = self.update(self.cmd.left, self.left_target)
            self.cmd.right = self.update(self.cmd.right, self.right_target)
            self.cmd_pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('smoother')
    S = Smooth()
    S.run()

#!/usr/bin/env python

import rospy
import numpy as np
from dreamer2ros.msg import Act2
from heron_msgs.msg import Twist

class Smooth:
    def __init__(self):
        self.cmd_pub = rospy.Publisher('cmd_drive', Twist, queue_size=1)
        rospy.Subscriber('cmd_rl', Act2, self.cmdCallback)
        self.rate = rospy.get_param('~rate',50)
        self.beta = rospy.get_param('~beta',0.2)
        self.wheel_offset = rospy.get_param('~wheel_offset') # offset for each wheel from front axis in meters
        self.lin_x_target = 0.
        self.ang_z_target = 0.
        self.cmd = Twist()

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

    def map_cmd(self, left, right):
        rotation = (right - left)/(2*self.wheel_offset)
        linear   = (left + right)/2
        return rotation, linear
        
    def cmdCallback(self, data):
        left = self.check_and_saturate(data.a0)
        right = self.check_and_saturate(data.a1)
        self.ang_z_target, self.lin_x_target = self.map_cmd(left, right)

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
            self.cmd.linear.x = self.update(self.cmd.linear.x, self.lin_x_target)
            self.cmd.angular.z = self.update(self.cmd.angular.z, self.ang_z_target)
            self.cmd_pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('smoother')
    S = Smooth()
    S.run()

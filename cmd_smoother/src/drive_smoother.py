#!/usr/bin/env python

import rospy
import numpy as np
from dreamer2ros.msg import Act2
from heron_msgs.msg import Drive

class Smooth:
    def __init__(self):
        self.cmd_pub = rospy.Publisher('cmd_drive', Drive, queue_size=1)
        rospy.Subscriber('cmd_rl', Act2, self.cmdCallback)
        self.rate = rospy.get_param('~rate',50)
        self.beta = rospy.get_param('~beta',0.2)
        self.left_target = 0.
        self.right_target = 0.
        self.cmd = Drive()

    def cmdCallback(self, data):
        if np.isnan(data.a0):
            self.left_target  = 0
        elif data.a0 > 1.0:
            self.left_target = 1
        elif data.a0 < -1.0:
            self.left_target = -1
        else:
            self.left_target  = data.a0
        if np.isnan(data.a1):
            self.right_target = 0
        elif data.a1 > 1.0:
            self.right_target = 1.0
        elif data.a1 < -1.0:
            self.right_target = -1.0
        else:
            self.right_target = data.a1
        

    def run(self):
        rate = rospy.Rate(self.rate)
        self.cmd.left = 0.0
        self.cmd.right = 0.0
        while not rospy.is_shutdown():
            if abs(self.cmd.right - self.right_target) > self.beta:
                self.cmd.right = self.cmd.right + np.sign(- self.cmd.right + self.right_target)*self.beta
            else:
                self.cmd.right = self.right_target
            if abs(self.cmd.left - self.left_target) > self.beta:
                self.cmd.left = self.cmd.left + np.sign(-self.cmd.left + self.left_target)*self.beta
            else:
                self.cmd.left = self.left_target

            self.cmd_pub.publish(self.cmd)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('smoother')
    S = Smooth()
    S.run()

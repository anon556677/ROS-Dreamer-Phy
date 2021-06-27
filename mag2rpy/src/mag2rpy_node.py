#!/usr/bin/env python
from scipy import optimize
import numpy as np
import pickle
import rospy
import os

import tf
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Vector3Stamped
from visualization_msgs.msg import Marker
from mag2rpy.srv import startCallibration, endCallibration, stopCallibration
from mag2rpy.srv import startCallibrationResponse, endCallibrationResponse, stopCallibrationResponse

class mag2rpy_node:
    def __init__(self):
        self.mag_buffer = []
        self.acquire = False
        self.load_ok = False
        self.offset = np.zeros(2)

        self.path = rospy.get_param('~path2statistics','hihi')
        self.use_markers = rospy.get_param('~use_markers', False)
        self.invert_yaw = rospy.get_param('~invert_yaw', False)
        self.yaw_offset = rospy.get_param('~yaw_offset', 0.0)
        self.frame = rospy.get_param('~frame','base_link')

        self.rpy_ned = Vector3Stamped()
        self.rpy_ned.header.frame_id = self.frame
        self.rpy_enu = Vector3Stamped()
        self.rpy_enu.header.frame_id = self.frame

        self.marker_ned = Marker()
        self.marker_ned.header.frame_id = self.frame
        self.marker_ned.type = 0
        self.marker_ned.scale.x = 1
        self.marker_ned.scale.y = 0.1
        self.marker_ned.scale.z = 0.1
        self.marker_ned.color.a = 1.0
        self.marker_ned.color.r = 1.0
        self.marker_ned.color.g = 0.0
        self.marker_ned.color.b = 0.0
        self.marker_enu = Marker()
        self.marker_enu.header.frame_id = self.frame
        self.marker_enu.type = 0
        self.marker_enu.scale.x = 1
        self.marker_enu.scale.y = 0.1
        self.marker_enu.scale.z = 0.1
        self.marker_enu.color.a = 1.0
        self.marker_enu.color.r = 0.0
        self.marker_enu.color.g = 0.0
        self.marker_enu.color.b = 1.0

        self.loadOffsetStatistics()

        rospy.Service("~start", startCallibration, self.handleStartCallibration)
        rospy.Service("~end", endCallibration, self.handleEndCallibration)
        rospy.Service("~stop", stopCallibration, self.handleStopCallibration)

        self.rpy_ned_pub = rospy.Publisher("~rpy_ned", Vector3Stamped, queue_size=1)
        self.rpy_enu_pub = rospy.Publisher("~rpy_enu", Vector3Stamped, queue_size=1)
        self.marker_ned_pub = rospy.Publisher("~marker_ned", Marker, queue_size=1)
        self.marker_enu_pub = rospy.Publisher("~marker_enu", Marker, queue_size=1)

        rospy.Subscriber("~mag", Vector3Stamped, self.magCallback, queue_size=1)

    def  handleStartCallibration(self, req):
        if req.start:
            self.acquire = True
        return startCallibrationResponse(True)
    
    def f_2(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = self.calc_R(*c)
        return Ri - Ri.mean()

    def handleEndCallibration(self, req):
        self.acquire = False
        self.mag_buffer = np.array(self.mag_buffer)
        center_estimate = (np.mean(self.mag_buffer[:,0]), np.mean(self.mag_buffer[:,1]))
        center, _ = optimize.leastsq(self.f_2, center_estimate)
        self.mag_buffer = []
        data = {}
        if self.load_ok:
            self.old_offset = self.offset.copy()
            rospy.loginfo("Last offset: x: "+str(self.old_offset[0])+" y:"+str(self.old_offset[1]))
        else:
            self.offsets = []
        self.offset = np.array([center[0],center[1]])
        self.offsets.append(self.offset)
        data['offsets'] = self.offsets
        rospy.loginfo("New offset: x: "+str(self.offset[0])+" y: "+str(self.offset[1]))
        with open(self.path, 'wb') as handle:
            pickle.dump(data, handle)

        return endCallibrationResponse(self.offset[0],self.offset[1])

    def handleStopCallibration(self, req):
        if req.stop:
            self.acquire = False
            self.mag_buffer = []
        return stopCallibrationResponse(True)

    def loadOffsetStatistics(self):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as handle:
                data = pickle.load(handle)
            self.load_ok = True
            self.offset = data['offsets'][-1]
            self.offsets = data['offsets']
            offsets = np.array(self.offsets)
            rospy.loginfo("Previous calibrations loaded.")
            rospy.loginfo("x_offset: "+str(self.offset[0])+", y_offset:"+str(self.offset[1]))
            rospy.loginfo("std_dev_x: "+str(np.std(offsets[:,0]))+" ,std_dev_y: "+str(np.std(offsets[:,1])))

    def magCallback(self, msg):
        msg = msg.vector
        if self.acquire:
            self.mag_buffer.append([msg.x, msg.y])
        #yaw between -pi and pi
        yaw = np.fmod(np.arctan2(msg.y - self.offset[1], msg.x - self.offset[0]) + self.yaw_offset +3*np.pi,2*np.pi)-np.pi
        if self.invert_yaw:
            yaw = -yaw
        self.rpy_ned.header.stamp = rospy.Time.now()
        self.rpy_ned.vector.z = yaw
        self.rpy_ned_pub.publish(self.rpy_ned)
        # NED to ENU conversion : + PI/2 - YAW
        self.rpy_enu.header.stamp = rospy.Time.now()
        self.rpy_enu.vector.z = np.pi - yaw
        self.rpy_enu_pub.publish(self.rpy_enu)
        if self.use_markers:
            Q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            self.marker_ned.pose.orientation.x = Q[0]
            self.marker_ned.pose.orientation.y = Q[1]
            self.marker_ned.pose.orientation.z = Q[2] 
            self.marker_ned.pose.orientation.w = Q[3] 
            # NED to ENU conversion : + PI/2 - YAW
            Q = tf.transformations.quaternion_from_euler(0, 0, np.pi/2 - yaw)
            self.marker_enu.pose.orientation.x = Q[0]
            self.marker_enu.pose.orientation.y = Q[1]
            self.marker_enu.pose.orientation.z = Q[2] 
            self.marker_enu.pose.orientation.w = Q[3] 
            self.marker_ned_pub.publish(self.marker_ned)
            self.marker_enu_pub.publish(self.marker_enu)
            
    def calc_R(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.mag_buffer[:,0]-xc)**2 + (self.mag_buffer[:,1]-yc)**2)


if __name__ == "__main__":
    rospy.init_node('mag2rpy')
    m2r = mag2rpy_node()
    rospy.spin()

#!/usr/bin/python
import os
#we need that to avoid numpy calling openBLAS which is highly CPU extensive...
os.environ["OMP_NUM_THREADS"] = "1"
import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Vector3Stamped
from heron_msgs.msg import Drive
import tf
import copy
import utm
import numpy as np
from math import cos, sin, atan2, pi, sqrt, hypot, fabs
from ekf_plot import PlotEKF
import matplotlib.pyplot as plt

class EKFLocalization:
    def __init__(self):
        #class members, initialization and flags
        self.rpy_imu = Vector3Stamped()
        self.rpy_rtk = Vector3Stamped()
        self.imu = Imu()
        self.rtk = NavSatFix()
        self.new_rtk = False
        self.new_imu = False
        self.new_rpy_imu = False
        self.new_rpy_rtk = False
        self.prev_cmd = None
        self.prev_imu = None
        self.prev_rpy_imu = None
        self.prev_rpy_rtk = None
        self.prev_rtk = None
        self.cmd = None
        self.prev_time = None
        self.init_speed = None
        rospy.init_node('kf_ekf_loc')
        self.robot_frame = rospy.get_param("~robot_frame","base_link")
        self.odom_frame = rospy.get_param("~odom_frame","odom_ekf")
        self.gps_frame = rospy.get_param("~gps_frame","gps")#for the static transform between the position of the gps and base_link
        self.odom_as_parent = rospy.get_param("~odom_as_parent",False)#whether we publish the new odom frame as parent of base_link, default: as child
        self.DEBUG = rospy.get_param("~DEBUG",False)
        print(self.robot_frame, self.odom_frame, self.gps_frame) 
        #Constants we use to estimate the speed of the robot from the Drive message
        self.dist_motors = rospy.get_param("~dist_motors",0.75)
        self.max_vang = rospy.get_param("~max_vang",1.5)
        self.vmax = rospy.get_param("~vmax",2.0)
        self.vrear_max = rospy.get_param("~vrear_max",1.0)
        #subscribers and publishers
        self.imu_sub = rospy.Subscriber('imu', Imu, self.imuCallback, queue_size=1)
        self.rpy_imu_sub = rospy.Subscriber('rpy_imu', Vector3Stamped, self.rpyImuCallback, queue_size=1)
        self.rpy_rtk_sub = rospy.Subscriber('rpy_rtk', Vector3Stamped, self.rpyRtkCallback, queue_size=1)
        self.rtk_sub = rospy.Subscriber('rtk', NavSatFix, self.rtkCallback, queue_size=1)
        self.fuse_odom_pub = rospy.Publisher('~fused_odom', Odometry, queue_size=1)
        self.speed_pub = rospy.Publisher('~robot_speed', Odometry, queue_size=1)
        self.broadcaster = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.wait_for_init()
        #we wait for the first messages to initialize the state:
        self.rtk_init = self.get_baselink_position_from_rtk_in_utm(self.rtk)
        rpy_init = self.rpy_enu_from_ned(self.rpy_rtk.vector)[2]+pi/2
        #class members representing the state
        #we initialize the state with uncertainty 
        #state in 2d containing 7 variables: X, Y , YAW , VX, VY, VANG, + BIAS_GYRO
        self.X = np.vstack((0,0,rpy_init,self.init_speed[0],self.init_speed[1],self.imu.angular_velocity.z,0))
        #self.P = np.diag((1e-2,1e-2,0.1,1e-2, 1e-2, 1e-2,1e-8))
        self.P = np.diag((1e-2,1e-2,1e-2,1e-2, 1e-2, 1e-2,1e-2))
        #self.Q = np.diag((1e-4,1e-4,1e-4,1e-4,1e-4,1e-3,1e-7))
        self.Q = np.diag((1e-3,1e-3,1e-4,1e-4,1e-4,1e-3,1e-7))
        #Now that everything is ready, we can call the Callback on the wheel odometry:
        #self.cmd_drive_sub = rospy.Subscriber('cmd_drive', Drive, self.cmdCallback, queue_size=1)
        rospy.loginfo("initialization complete, Initial State is:" + str(self.X))
        #allow to plot debug stuff for the EKF , to trigger:
        #rostopic pub -1 /end_plot std_msgs/Bool True
        if self.DEBUG:
            self.end_plot_sub = rospy.Subscriber('end_plot', Bool, self.endPlotCallback, queue_size=1)
            self.cmd = []
            self.end_plot = False
            self.X_list = []
            self.P_list = []
            self.A_list = []
            self.B_list = []
            self.biais_gyro = []
            self.cmd_list = []
            self.obs_rtk = []
            self.obs_imu = []
            self.obs_rpy_imu = []
            self.obs_rpy_rtk = []
            self.robot_speeds = []
            self.imu_corrections = []
            self.rpy_imu_corrections = []
            self.rpy_rtk_corrections = []
            self.rtk_corrections = []

    def wait_for_init(self):
        rospy.loginfo("Waiting for first rtk, imu and first speed")
        rospy.sleep(.5) 
        while not rospy.is_shutdown():
            if ((self.rpy_rtk == None) or (self.rtk == None)) or (self.imu == None):
                print("at least one None")
                continue
            else:
                print("nothing is None anymore,will try to init speed")
                if not self.can_init_speed():
                    print("speed not calculated yet")
                    continue
                else: 
                    print("speed calculated, init complete")
                    return
    
    def can_init_speed(self):
        if self.prev_rtk == None:
            self.prev_rtk = self.rtk
            return False
        dt = (self.rtk.header.stamp - self.prev_rtk.header.stamp).to_sec()
        print("in init_speed dt: ", dt)
        #convert the rtk lat,long into utm and into base_link
        [x,y,_] = self.get_baselink_position_from_rtk_in_utm(self.rtk)
        [x_prev,y_prev,_] = self.get_baselink_position_from_rtk_in_utm(self.prev_rtk)
        #print(x,y,x_prev,y_prev, dt)
        if not (dt < 1e-5):#(==0)
            print("init speed:" ,x,y,x_prev,y_prev, dt)
            vx = (x-x_prev)/dt 
            vy = (y-y_prev)/dt
            self.init_speed = [vx,vy]
            return True
        else: return False

    #prediction step of the EKF
    def predict_without_cmd(self):    
        if self.prev_time == None:
            self.prev_time = rospy.Time.now()
        dt = (rospy.Time.now()- self.prev_time).to_sec()
        self.prev_time = rospy.Time.now()
        #predict next state
        X_pred = np.array(([self.X[0,0] + self.X[3,0]*dt],
                            [self.X[1,0] + self.X[4,0]*dt],
                            [self.X[2,0] + self.X[5,0]*dt],
                            [self.X[3,0]],
                            [self.X[4,0]],
                            [self.X[5,0]],
                            [self.X[6,0]]))
        A = np.array(([1, 0, 0, dt,  0,  0,  0],
                        [0, 1, 0,  0, dt,  0, 0],
                        [0, 0, 1,  0,  0, dt, 0],
                        [0, 0, 0,  1,  0,  0, 0],
                        [0, 0, 0,  0,  1,  0, 0],
                        [0, 0, 0,  0,  0,  1, 0],
                        [0, 0, 0,  0,  0,  0, 1]))
        P_pred = np.matmul(np.matmul(A, self.P), np.transpose(A)) + self.Q
        return X_pred, P_pred
    
    #prediction step of the EKF
    def predict(self, msg):    
        if self.prev_cmd == None:
            self.prev_cmd = msg
        if self.prev_time == None:
            self.prev_time = rospy.Time.now()
        dt = (self.prev_time - rospy.Time.now()).to_sec()
        self.prev_time = rospy.Time.now()
        dvlin, dvang = self.get_dv_from_cmd_drive() 
        self.cmd_list.append([dvlin,dvang,dt])    
        #predict next state
        X_pred = np.array(([self.X[0,0] + (dvlin + hypot(self.X[3,0], self.X[4,0]))*dt*cos(self.X[2,0])],
                            [self.X[1,0] + (dvlin + hypot(self.X[3,0], self.X[4,0]))*dt*sin(self.X[2,0])],
                            [np.fmod(self.X[2,0] + (dvang + self.X[5,0])*dt + 3*pi, 2*pi) - pi],
                            #[self.X[2,0]],
                            [self.X[3,0]+dvlin*cos(self.X[2,0])],
                            [self.X[4,0]+dvlin*sin(self.X[2,0])],
                            [self.X[5,0]+dvang],
                            [self.X[6,0]]))
        A = self.get_jacobian_state(dvlin, dvang, dt)
        B = self.get_jacobian_command(dvlin, dvang, dt)
        #A = np.identity(7)
        #B = np.zeros((7,2))   
        Qu = np.diag(((.5)**2, (1)**2))
        a = np.matmul(np.matmul(A, self.P), np.transpose(A))
        b= np.matmul(np.matmul(B, Qu), np.transpose(B))
        self.A_list.append(copy.deepcopy(a))
        self.B_list.append(copy.deepcopy(b))
        P_pred = np.matmul(np.matmul(A, self.P), np.transpose(A)) + np.matmul(np.matmul(B, Qu), np.transpose(B)) + self.Q
        return X_pred, P_pred
    
    #correction step based on the last sensors readings:
    def correct(self, X_pred, P_pred):
        Zk = np.empty([0,1])
        R_diag = np.empty([0,1])
        H = np.empty([0,7])
        #we correct based on which new sensor reading we have
        #observation of the angular speed from the imu
        if self.new_imu:
            #IMU is pointing down, we need to invert the sign
            Zk = np.vstack((Zk,-self.imu.angular_velocity.z))
            R_diag = np.vstack((R_diag, 0.035**2))#/5
            H = np.vstack((H,(0,0,0,0,0,1,1)))
            if self.DEBUG:
                self.obs_imu.append(-self.imu.angular_velocity.z)
                h = np.array((0,0,0,0,0,1,1))
                z = np.vstack([-self.imu.angular_velocity.z])
                c = z - np.matmul(h, X_pred)
                self.imu_corrections.append(c.tolist())
        #observation of the yaw from the imu
        if self.new_rpy_imu:
            current_in_enu = self.rpy_enu_from_ned(self.rpy_imu.vector)[2] +pi/2
            Zk = np.vstack((Zk,current_in_enu))
            R_diag = np.vstack((R_diag, 0.25**2))#/5
            H = np.vstack((H,(0,0,1,0,0,0,0)))
            if self.DEBUG:
                self.obs_rpy_imu.append(current_in_enu)
                h = np.array((0,0,1,0,0,0,0))
                z = np.vstack([current_in_enu])
                c = z - np.matmul(h,X_pred)
                self.rpy_imu_corrections.append(c.tolist())
        #observation of the position from the rtk
        if self.new_rtk:
            #convert the rtk lat,long into utm and into base_link
            [x, y, _] = self.get_baselink_position_from_rtk_in_utm(self.rtk)
            Zk = np.vstack((Zk, x-self.rtk_init[0], y-self.rtk_init[1]))
            if (self.rtk.status.status == 3):  # fix, we are confident
                R_diag = np.vstack((R_diag, 0.15**2, 0.15**2))
            elif (self.rtk.status.status == 2):  # float, we are  less confident
                R_diag = np.vstack((R_diag, 1**2, 1**2))
            else:  # bad, we don't trust it
                R_diag = np.vstack((R_diag, 5**2, 5**2))
            H = np.vstack((H, (1,0,0,0,0,0,0), (0,1,0,0,0,0,0)))
            if self.DEBUG:
                self.obs_rtk.append([x-self.rtk_init[0], y-self.rtk_init[1]])
                h = np.vstack(((1,0,0,0,0,0,0),(0,1,0,0,0,0,0)))
                z = np.vstack((x-self.rtk_init[0], y-self.rtk_init[1]))
                c = z - np.matmul(h, X_pred)
                self.rtk_corrections.append(c.tolist())
        #now, do the actual correction based on these observations
        R = np.diagflat(R_diag)
        K = np.matmul(np.matmul(P_pred, np.transpose(H)), np.linalg.inv(np.matmul(H, np.matmul(P_pred, np.transpose(H)))+R))
        self.P = copy.deepcopy(np.matmul(np.identity(7) - np.matmul(K, H), P_pred))
        #self.X = copy.deepcopy(X_pred + np.matmul(K, (Zk - np.matmul(H, X_pred))))
        Zi = Zk - np.matmul(H, X_pred)
        #if we update the yaw, make sure we deal with 2pi jumps
        if H.size != 0:
            if np.sum(H,axis=0)[2]:
                i = np.argwhere(H[:,2]==1)
                Zi[i,0] = np.fmod(Zi[i,0]+3*pi,2*pi)-pi
        self.X = copy.deepcopy(X_pred + np.matmul(K, Zi))
        
        #set back the flags to False
        self.new_imu = False
        self.new_rtk = False
        self.new_rpy_imu = False
        self.new_rpy_rtk = False
    
    #utility function to get the displacement from the cmd
    def v_from_alpha(self,alpha):
        if alpha >=0 :
            return alpha * self.vmax
        else:
            return alpha * self.vrear_max
    
    #utility function to get the displacement from the cmd
    def get_twist_from_cmd_drive(self, cmd):
        vleft = self.v_from_alpha(cmd.left)
        vright = self.v_from_alpha(cmd.right)
        vlin = (vleft+vright)/2
        vang = min(max((vright-vleft)/(self.dist_motors/2.0),-self.max_vang),self.max_vang)
        return [vlin, vang]

    #function to get the displacement from the cmd
    def get_dv_from_cmd_drive(self):
        twist_curr = self.get_twist_from_cmd_drive(self.cmd)
        twist_prev = self.get_twist_from_cmd_drive(self.prev_cmd)
        dvlin = twist_curr[0] - twist_prev[0]
        dvang = twist_curr[1] - twist_prev[1]
        return dvlin, dvang

    #function to get position of the rtk from lat,long to the base_link in meters
    def get_baselink_position_from_rtk_in_utm(self, rtk):
        x,y = utm.from_latlon(rtk.latitude, rtk.longitude)[:2]
        z = rtk.altitude
        if self.listener.canTransform(self.robot_frame, self.gps_frame, rospy.Time(0)):
            (trans,quat) = self.listener.lookupTransform(self.robot_frame, self.gps_frame, rospy.Time(0))
            x += trans[0]
            y += trans[1]
            z += trans[2]
            return [x,y,z]
        else:
            return[x,y,z]

    #convert the imu from enu to ned
    def rpy_enu_from_ned(self, vector):
        return [vector.x, -vector.y, pi/2-vector.z]

    #sensors callback
    def cmdCallback(self, msg):
        print(msg)
        self.cmd = msg
        #predict and correct
        X_pred, P_pred = self.predict(msg)
        self.correct(X_pred, P_pred)
        if self.DEBUG:
            self.X_list.append(copy.deepcopy(self.X))
            self.P_list.append(copy.deepcopy(self.P))
            self.biais_gyro.append(copy.deepcopy(self.X[6,0]))
        #publish an Odometry
        self.publish_odom()
        self.publish_speeds()
    
    def imuCallback(self, msg):
        self.imu = msg
        self.new_imu = True
    
    def rpyImuCallback(self, msg):
        self.rpy_imu = msg
        self.new_rpy_imu = True
    
    def rpyRtkCallback(self, msg):
        self.rpy_rtk = msg
        self.new_rpy_rtk = True

    def rtkCallback(self, msg):
        self.rtk = msg
        self.new_rtk = True

    def endPlotCallback(self, msg):
        self.end_plot = msg
   
   #publishes an Odometry and broadcast the transform 
    def publish_odom(self):
        msg = Odometry()
        msg.header.frame_id = self.odom_frame
        msg.header.stamp = rospy.Time.now()
        translation = (self.X[0,0], self.X[1,0], self.X[2,0])
        msg.pose.pose.position.x = translation[0] 
        msg.pose.pose.position.y = translation[1] 
        msg.pose.pose.position.z = translation[2] 
        quat = tf.transformations.quaternion_from_euler(0,0,self.X[2,0])
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        #visualize the ellipses
        cov = np.zeros((6,6))
        cov = copy.deepcopy(self.P[:6,:6])
        msg.pose.covariance = cov.flatten().tolist()        

        if not self.odom_as_parent: 
            #we want to publish the tf as child of base_link, because we have multiple odom frames:
            #we broadcast the inverse transform
            transform = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(translation), tf.transformations.quaternion_matrix(quat))
            inverse = tf.transformations.inverse_matrix(transform)
            t = tf.transformations.translation_from_matrix(inverse) 
            q = tf.transformations.quaternion_from_matrix(inverse)
            self.broadcaster.sendTransform(t, q, msg.header.stamp, self.odom_frame, self.robot_frame)
        else: 
            ##we broadcast the transform with odom as a parent of the robot_frame:
            self.broadcaster.sendTransform(translation, quat, msg.header.stamp, self.robot_frame, self.odom_frame)
        
        self.fuse_odom_pub.publish(msg)

    
    #publishes only the speeds in the robot_frame (as an Odom)    
    def publish_speeds(self):
        #we transform the speed from odom into the robot frame
        #rot = np.array(((cos(self.X[2,0]), -sin(self.X[2,0])), (sin(self.X[2,0]), cos(self.X[2,0]))))
        #v_odom = np.vstack((self.X[3,0], self.X[4,0]))
        #v_robot = np.matmul(rot, v_odom)
        vlin = self.X[3,0] * cos(self.X[2,0]) + self.X[4,0]*sin(self.X[2,0])
        vtrans = - self.X[3,0] * sin(self.X[2,0]) + self.X[4,0]*cos(self.X[2,0])
        
        msg = Odometry()
        msg.header.frame_id = self.robot_frame
        msg.header.stamp = rospy.Time.now()
        msg.twist.twist.linear.x = vlin
        msg.twist.twist.linear.y = vtrans
        msg.twist.twist.angular.z = self.X[5,0]
        self.speed_pub.publish(msg)
        if self.DEBUG:
            self.robot_speeds.append([vlin, vtrans, self.X[5,0]])

        
        #if self.listener.canTransform(self.robot_frame, self.odom_frame, rospy.Time(0)):
        #    v = Vector3Stamped()
        #    v.header.stamp= rospy.Time(0)
        #    v.header.frame_id= self.odom_frame
        #    v.vector.x = 0
        #    v.vector.y = 0
        #    v.vector.z = self.X[2,0]
        #    #vr = tf.TransformerROS.transformVector3(self.robot_frame, v) 
        #    vr = self.listener.transformVector3(self.robot_frame, v) 
        #    msg = Odometry()
        #    msg.header.frame_id = self.robot_frame
        #    msg.header.stamp = rospy.Time.now()
        #    msg.twist.twist.linear.x = vr.vector.x
        #    msg.twist.twist.linear.y = vr.vector.y
        #    msg.twist.twist.angular.z = self.X[5,0]
        #    self.speed_pub.publish(msg)
        #    if self.DEBUG:
        #        self.robot_speeds.append([vr.vector.x, vr.vector.y, self.X[5,0]])

    #Calculate the jacobians
    def get_jacobian_state(self, dvlin, dvang, dt):
        THETA = self.X[2,0]
        VX = self.X[3,0]
        VY = self.X[4,0]
        x0 = sin(THETA);
        x1 = sqrt(pow(VX, 2) + pow(VY, 2));
        x2 = dt*(dvlin + x1);
        x3 = cos(THETA);
        #avoid dividing by 0:
        if x1 == 0:
            x4 = 0
        else:
            x4 = dt/x1;
        x5 = x3*x4;
        x6 = x0*x4;
        A = np.array([[1, 0,    -x0*x2, VX*x5, VY*x5,  0,0],
                    [0, 1,     x2*x3, VX*x6, VY*x6,  0,0],
                    [0, 0,         1,     0,     0, dt,0],
                    [0, 0, -dvlin*x0,     1,     0,  0,0],
                    [0, 0,  dvlin*x3,     0,     1,  0,0],
                    [0, 0,         0,     0,     0,  1,0],
                    [0, 0,         0,     0,     0,  0,1]])
        return A


    def get_jacobian_command(self, dvlin, dvang, dt):
        THETA = self.X[2,0]
        x0 = cos(THETA);
        x1 = sin(THETA);
        B = np.array([[dt*x0,  0],
                    [dt*x1,  0],
                    [    0, dt],
                    [   x0,  0],
                    [   x1,  0],
                    [    0,  1],
                    [    0,  0]])
        return B

    def plot(self):
        p_rtk = []
        for obs in self.obs_rtk:
            p_rtk.append([obs[0], obs[1]])


        yaws = []
        ang_speed = []
        v = []
        for state in self.X_list:
            yaws.append(state[2,0])
            ang_speed.append(state[5,0])
            v.append([state[3,0],state[4,0]])

        self.robot_speeds = np.array(self.robot_speeds)

        fig, axs = plt.subplots(3, 1)
        fig.set_figheight(11)
        fig.set_figwidth(11)
        axs[0].plot(self.robot_speeds[:,0],label="vlin")
        axs[0].plot([0,self.robot_speeds[:,0].shape[0]],[0,0],'k--')
        axs[1].plot(self.robot_speeds[:,1],label="vtrans")
        axs[1].plot([0,self.robot_speeds[:,1].shape[0]],[0,0],'k--')
        axs[2].plot(self.robot_speeds[:,2],label="vang")
        axs[2].plot([0,self.robot_speeds[:,2].shape[0]],[0,0],'k--')
        axs[0].legend(loc="upper right", shadow = True, fontsize = "x-small")
        axs[1].legend(loc="upper right", shadow = True, fontsize = "x-small")
        axs[2].legend(loc="upper right", shadow = True, fontsize = "x-small")
        fig.savefig(os.path.join(os.getcwd(), "speeds.png"))

        PlotEKF(([[0,0], self.X_list, "state_6",["X", "Y", "THETA", "VX", "VY", "VANG"]],
               [[0,3], self.P_list, "diagonal_6",["Pxx", "Pyy", "Ptt", "Pvxvx", "Pvyvy", "Pvangvang"]],
               [[1,0], yaws, "list",["STATE_THETA"]],
               [[2,0], ang_speed, "list",["STATE_ANG_SPEED"]],
               [[3,0], v, "list",["STATE_VX","STATE_VY"]],
               [[2,1], self.imu_corrections, "list",["correction-imu"]],
               [[1,1], self.rpy_imu_corrections, "list",["correction-imu-yaw"]],
               [[3,1], self.rpy_rtk_corrections, "list",["correction-rtk-yaw"]],
               [[0,1], self.rtk_corrections, "list",["correction-rtk-x","correction-rtk-y"]],
               [[0,2], p_rtk, "list",["rtkx", "rtky"]],
               [[1,2], self.obs_rpy_imu, "list",["yaw_imu"]],
               [[3,2], self.obs_rpy_rtk, "list",["yaw_rtk"]],
               [[1,3], self.robot_speeds, "list",["vlin_robot","vtrans_robot","vang_robot"]],
               [[2,2], self.obs_imu, "list",["imu-ang_speed"]],
               [[2,3], self.biais_gyro, "list",["biais-gyro"]]),
               plot_list=True, plot_xyz=True, planar=True, anim_ellipse=False)      


    #keep the node alive
    def run(self):
        rate = rospy.Rate(20)
        t_init = rospy.Time.now()
        while not rospy.is_shutdown():
            #predict at 20Hz
            #predict and correct
            X_pred, P_pred = self.predict_without_cmd()
            self.correct(X_pred, P_pred)
            #self.X = copy.deepcopy(X_pred)
            #self.P = copy.deepcopy(P_pred)
            #publish an Odometry
            self.publish_odom()
            self.publish_speeds()
            duration = (rospy.Time.now() - t_init).to_sec()
            if self.DEBUG:
                self.X_list.append(copy.deepcopy(self.X))
                self.P_list.append(copy.deepcopy(self.P))
                self.biais_gyro.append(copy.deepcopy(self.X[6,0]))
                if self.end_plot or duration >= 500:
                    print("WILL PLOT")
                    self.plot()
                    self.end_plot = False
                    break
            rate.sleep()

if __name__ == '__main__':
    loc = EKFLocalization()
    loc.run()






#!/usr/bin/env python

import rospy
from bottle import Bottle, request
import json
import numpy as np
from std_msgs.msg import Bool 
from rl_server.msg import Episode
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from uuv_world_ros_plugins_msgs.srv import SetCurrentVelocity
from uuv_gazebo_ros_plugins_msgs.srv import SetFloat
from dreamer2ros.msg import Act2

import sampling_helper as sh

class Server:
    def __init__(self):
        # Publishers
        self.episode_manager_pub_ = rospy.Publisher('server/episode_manager', Episode, queue_size=1)  
        self.action_pub_ = rospy.Publisher('cmd_rl', Act2, queue_size=1)
        self.sim_ok_pub_ = rospy.Publisher('server/sim_ok', Bool, queue_size=1)  
        # Subscribers
        rospy.Subscriber('agent/is_done', Bool, self.doneCallback)
        # Services
        self.spawn_service_ = rospy.get_param('~spawn_service','')
        # Server settings
        self.port_ = rospy.get_param('~server_port',8080)
        # Curriculum settings
        self.mode_ = rospy.get_param('~curriculum',"none")
        max_dist = rospy.get_param('~max_dist', 5.0)
        min_dist = rospy.get_param('~min_dist', 9.0)
        boey_dist = rospy.get_param('~boey_dist', 2.5)
        usv_view = rospy.get_param('~usv_view', 110)
        boey_cone = rospy.get_param('~boey_cone', 110)
        alpha = rospy.get_param('~alpha', 1.)
        warmup = rospy.get_param('~warmup', 0.25e6)
        target_step = rospy.get_param('~target_step', 1.0e6)
        # Others
        self.robot_name_ = rospy.get_param('~robot_name','')
        self.boey_name_ = rospy.get_param('~boey_name','')
        self.path_to_data_ = rospy.get_param('~path_to_pose','blabla')
        # Reset command
        self.make_reset_cmd()
        # Spawn
        self.robot_spawn_req = self.make_spawn_req(self.robot_name_)
        self.boey_spawn_req = self.make_spawn_req(self.boey_name_)
        # Bridge server
        self.makeServer()
        # Pose sampler
        self.PS = sh.ParkingSampler(self.path_to_data_, buoy_dist=boey_dist,\
                                     usv_spawn_cone=boey_cone,\
                                     usv_view_angle=usv_view,\
                                     min_buoy_usv=min_dist,\
                                     max_buoy_usv=max_dist,\
                                     target_step=target_step,\
                                     warmup=warmup,\
                                     alpha=alpha)
    
    def makeServer(self):
        self._host = 'localhost'
        self._app = Bottle()
        self._route()
        self.expected_keys_ = ['random', 'steps', 'repeat', 'discount', 'training', 'current_step', 'reward']
        self.check_rate_ = 2.0
        self.op_OK_ = False
    
    def make_reset_cmd(self):
        self.reset_cmd = Act2()
        self.reset_cmd.a0 = 0
        self.reset_cmd.a1 = 0

    def make_spawn_req(self, name):
        # SPAWN SERVICE
        pose_req_ = ModelState()
        pose_req_.model_name = name
        pose_req_.pose.position.x = 0
        pose_req_.pose.position.y = 0
        pose_req_.pose.position.z = 0.145
        pose_req_.pose.orientation.x = 0
        pose_req_.pose.orientation.y = 0
        pose_req_.pose.orientation.z = 0
        pose_req_.pose.orientation.w = 0
        pose_req_.twist.linear.x = 0
        pose_req_.twist.linear.y = 0
        pose_req_.twist.linear.z = 0
        pose_req_.twist.angular.x = 0
        pose_req_.twist.angular.y = 0
        pose_req_.twist.angular.z = 0
        return pose_req_

    def _route(self):
        self._app.route('/toServer', method="POST", callback=self._onPost)
 
    def start(self):
        self._app.run(host=self._host, port=self.port_, reloarder=False)

    def doneCallback(self, msg):
        if msg.data:
            self.op_OK_=True

    def _onPost(self):
        req = json.loads(request.body.read())
        print(req)
        ep = Episode()
        for i in req.keys():
            if i  not in self.expected_keys_:
                raise ValueError('incorrect post request. You must provide the following fields:'+str(self.expected_keys_))
        ep.steps = req['steps']
        ep.random_agent = (req['random'] == 1)
        ep.discount = req['discount']
        ep.training = (req['training'] == 1)
        rospy.wait_for_service(self.spawn_service_)
        for i in range(req['repeat']+1):
            # Set wait for agent to say it's done
            self.op_OK_ = False
            # Sample position
            boey, usv, t, pdf = self.PS.sampleBuoyAndUSV(req['current_step'], self.mode_, req['reward'])
            self.spawnObject(self.boey_spawn_req, boey)
            self.spawnObject(self.robot_spawn_req, usv)
            # Set Domain Randomization (if any)
            # Give the agent the new mission parameters
            self.episode_manager_pub_.publish(ep)
            rospy.sleep(1.0)
            # Tell the agent it's good to go
            self.sim_ok_pub_.publish(True)
            # Wait for the agent to be done
            while ((not self.op_OK_) and (not rospy.is_shutdown())):
                rospy.sleep(self.check_rate_)
            # Reset sim
            print("Restoring environment...")
            print("Enforcing null commands...")
            self.action_pub_.publish(self.reset_cmd)
            print("Zero velocity refresh...")
            self.spawnObject(self.robot_spawn_req, usv)
        return 'Done'

    def spawnObject(self, req, pose):
        try:
            set_state = rospy.ServiceProxy(self.spawn_service_, SetModelState)
            req.pose.position.x = pose[0]
            req.pose.position.y = pose[1]
            q = sh.euler2quat(pose[2])
            req.pose.orientation.x = q[0]
            req.pose.orientation.y = q[1]
            req.pose.orientation.z = q[2]
            req.pose.orientation.w = q[3]      
            resp = set_state(req)
            print("refresh Ok new boat pose: x:", pose[0]," y:",pose[1]," heading:",pose[2])
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        rospy.sleep(1.0)

if __name__ == "__main__":
    rospy.init_node('server')
    server = Server()
    server.start()

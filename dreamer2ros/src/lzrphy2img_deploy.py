#!/usr/bin/env python

import rospy

import gym
import uuid
import io
import pathlib
import datetime
import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

from std_msgs.msg import Float32, Bool
from rl_server.msg import Episode
from dreamer2ros.msg import Act2
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import TwistStamped

import laser_phy_policy as laser_policy
import tools

class DreamerAgent:
    def __init__(self):
        # DREAMER
        self.agent = None
        self.save_directory = rospy.get_param('~model_path','')
        self.Done = True
        self.obs = {}
        self.precision = 32
        self.max_steps = 1000
        self.velocity = np.zeros(3)
        # ROS
        self.initialize_agent()
        self.refresh_agent()
        self.act = Act2()
        self.action_pub_ = rospy.Publisher('cmd_rl', Act2, queue_size=1)
        rospy.Subscriber("laser/scan", LaserScan, self.laserCallback, queue_size=1)
        rospy.Subscriber("reward_generator/DreamersView", Image, self.imageCallback, queue_size=1)
        rospy.Subscriber("vel", TwistStamped, self.velCallback, queue_size=1)

    def initialize_agent(self):
        parser = argparse.ArgumentParser()
        for key, value in laser_policy.define_config().items():
            parser.add_argument('--'+str(key), type=tools.args_type(value), default=value)
        config, unknown = parser.parse_known_args()
        if config.gpu_growth:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        assert config.precision in (16, 32), config.precision
        if config.precision == 16:
            prec.set_policy(prec.Policy('mixed_float16'))
        config.steps = int(config.steps)

        actspace = gym.spaces.Box(np.array([-1,-1]),np.array([1,1]))
        self.agent = laser_policy.Dreamer(config, actspace)
        if pathlib.Path(self.save_directory).exists():
            print('Load checkpoint.')
            self.agent.load(self.save_directory)
        else:
            raise ValueError('Could not load weights')
        self.state = None

    def refresh_agent(self):
        self.env_state = None
        self.phy_state = None
        self.step = 0
        if pathlib.Path(self.save_directory).exists():
            print('Load checkpoint.')
            self.agent.load(self.save_directory)
        else:
            raise ValueError('Could not load weights')
        self.obs['laser'] = np.zeros((1,256,1),dtype=np.float32)
        self.obs['image'] = np.zeros((1,64,64,3),dtype=np.uint8)
        self.obs['physics'] = np.zeros((1,3))
        self.obs['physics_d'] = np.zeros((1,3))
        self.Done = False

    def imageCallback(self, obs):
        self.image = np.reshape(np.fromstring(obs.data, np.uint8),[64,64,3])
        if not self.Done:
            self.obs['laser'][0] = self.laser
            self.obs['image'][0] = self.image
            self.obs['physics'][0] = self.velocity
            t_actions, self.env_state, self.phy_state = self.agent.policy(self.obs, self.env_state, self.phy_state, False)
            actions = np.array(t_actions)[0]
            self.action_pub_.publish(self.actionsConverter(actions))
            # Time-shift
            self.actions = actions
            self.obs['physics_d'][0] = self.obs['physics'][0].copy()
    
    def laserCallback(self, obs):
        ranges = np.array(obs.ranges)
        ranges[ranges==0] = 100000
        self.laser = np.expand_dims(np.clip(np.min(np.reshape(np.nan_to_num(ranges),[-1,2]),axis=1)[-256:], 0, 100000),-1)
    
    def actionsConverter(self, actions):
        self.act.a0 = actions[0]
        self.act.a1 = actions[1]
        return self.act
    
    def velCallback(self, vel):
        if not self.Done:
            self.velocity[0] = vel.twist.linear.x
            self.velocity[1] = vel.twist.linear.y
            self.velocity[2] = vel.twist.angular.z
         
if __name__ == "__main__":
    rospy.init_node('dreamer_agent')
    DA = DreamerAgent()
    rospy.spin()


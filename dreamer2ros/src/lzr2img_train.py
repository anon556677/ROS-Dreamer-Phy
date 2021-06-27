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

import laser_policy
import tools

class DreamerAgent:
    def __init__(self):
        # DREAMER
        self.agent = None
        self.save_directory = rospy.get_param('~model_path','')
        self.episode = {}
        self.Done = True
        self.random_agent = True
        self.reset = False
        self.agent_not_initialized = True
        self.setup_ok = False
        self.reset_agent_call = np.array([False])
        self.precision = 32
        self.max_steps = 1000
        self.obs = {}
        self.skip = 0
        self.obs['laser'] = np.zeros((1,256,1),dtype=np.uint8)
        self.obs['image'] = np.zeros((1,64,64,3),dtype=np.uint8)
        self.obs['reward'] = np.zeros((1))
        self.obs['physics'] = np.zeros((3))
        self.obs['physics_d'] = np.zeros((3))

        # ROS
        self.act = Act2()
        self.action_pub_ = rospy.Publisher('cmd_rl', Act2, queue_size=1)
        self.done_pub_ = rospy.Publisher('agent/is_done', Bool, queue_size=1)
        rospy.Subscriber("server/episode_manager", Episode, self.episodeCallback, queue_size=1)
        rospy.Subscriber("server/sim_ok", Bool, self.restartCallback, queue_size=1)
        rospy.Subscriber("front/scan", LaserScan, self.laserCallback, queue_size=1)
        rospy.Subscriber("reward_generator/reward", Float32, self.rewardCallback, queue_size=1)
        rospy.Subscriber("reward_generator/DreamersView", Image, self.imageCallback, queue_size=1)
        rospy.Subscriber("velocity_proj/robot_frame", TwistStamped, self.velCallback, queue_size=1)

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
        self.agent_not_initialized = False

    def refresh_agent(self):
        self.state = None
        self.step = 0
        self.episode = {}
        self.episode['physics'] = []
        self.episode['physics_d'] = []
        self.episode['image'] = []
        self.episode['laser'] = []
        self.episode['action'] = []
        self.episode['reward'] = []
        self.episode['discount'] = []
        if not self.random_agent:
            if pathlib.Path(self.save_directory).exists():
                print('Load checkpoint.')
                self.agent.load(self.save_directory)
            else:
                raise ValueError('Could not load weights')
        self.Done = True
        self.reset = True
        self.obs['laser'] = np.zeros((1,256,1),dtype=np.float32)
        self.obs['image'] = np.zeros((1,64,64,3),dtype=np.uint8)
        self.obs['reward'] = np.zeros((1))
        self.obs['physics'] = np.zeros((1,3))
        self.obs['physics_d'] = np.zeros((1,3))
        self.actions = np.zeros(2)
        self.reward = 0
        self.velocity = np.zeros(3)

    def imageCallback(self, obs):
        self.image = np.reshape(np.fromstring(obs.data, np.uint8),[64,64,3])
        if not self.Done:
            # Observe
            self.obs['laser'][0] = self.laser
            self.obs['image'][0] = self.image
            self.obs['reward'][0] = self.reward
            self.obs['physics'][0] = self.velocity
            # Play
            if self.random_agent:
                actions = (np.random.rand(2) - 0.5)*2
            else:
                t_actions, self.state = self.agent.policy(self.obs, self.state, self.training_mode)
                actions = np.array(t_actions)[0]
            self.action_pub_.publish(self.actionsConverter(actions))
            # Record
            self.incrementEpisode(self.obs['laser'][0], self.obs['image'][0], self.actions, self.obs['reward'][0], self.obs['physics'][0], self.obs['physics_d'][0])
            self.actions = actions
            self.obs['physics_d'][0] = self.obs['physics'][0].copy()
            #self.reward2 = self.obs['reward'][0]
            self.step += 1
            if self.step > self.max_steps:
                print('Done')
                self.Done = True
                self.saveEpisode()
                self.done_pub_.publish(True)
        else:
            self.skip += 1
                
        if self.reset:
            print('resetting and commencing new episode')
            self.reset = False
            self.Done = False
    
    def laserCallback(self, obs):
        self.laser = np.expand_dims(np.clip(np.min(np.reshape(np.nan_to_num(np.array(obs.ranges)),[-1,2]),axis=1)[-256:], 0, 100000),-1)
    
    def rewardCallback(self, reward):
        if not self.Done:
            self.reward = reward.data
    
    def velCallback(self, vel):
        if not self.Done:
            self.velocity[0] = vel.twist.linear.x
            self.velocity[1] = vel.twist.linear.y
            self.velocity[2] = vel.twist.angular.z

    def restartCallback(self, msg):
        if msg.data == True:
            print('sim is OK, waiting for agent...')
            while not(self.setup_ok):
                rospy.sleep(1)
            print('agent OK')
            self.setup_ok = False
            if ((not self.random_agent) and self.agent_not_initialized):
                self.initialize_agent()
            print('requesting refresh...')
            self.refresh_agent()
    
    def episodeCallback(self, msg):
        print('received new_episode settings')
        self.max_steps = msg.steps
        self.discount = msg.discount
        self.training_mode = msg.training
        self.random_agent = msg.random_agent
        self.setup_ok = True
        print('settings updated')

    def actionsConverter(self, actions):
        self.act.a0 = actions[0]
        self.act.a1 = actions[1]
        return self.act
         
    def incrementEpisode(self, laser, img, act, reward, phy, phy_d):
        self.episode['physics'].append(phy.copy())
        self.episode['physics_d'].append(phy_d.copy())
        self.episode['image'].append(img.copy())
        self.episode['laser'].append(laser.copy())
        self.episode['action'].append(act)
        self.episode['reward'].append(reward)
        self.episode['discount'].append(self.discount)
    
    def convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self.precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self.precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype = np.uint8
        else:
            raise NotImplementedError(value.dtype)
        return value.astype(dtype)

    def saveEpisode(self):
        if self.training_mode:
            directory = pathlib.Path(self.save_directory+'/episodes')
            try:
                directory.mkdir(parents=True)
            except:
                pass
        else:
            directory = pathlib.Path(self.save_directory+'/test_episodes')
            try:
                directory.mkdir(parents=True)
            except:
                pass
        self.episode = {k: self.convert(v) for k, v in self.episode.items()}
        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        #for episode in episodes:
        identifier = str(uuid.uuid4().hex)
        length = len(self.episode['reward'])
        filename = directory /'{}-{}-{}.npz'.format(timestamp,identifier,length)
        print('saving episode to '+str(filename))
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **self.episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())


if __name__ == "__main__":
    rospy.init_node('dreamer_agent')
    DA = DreamerAgent()
    rospy.spin()


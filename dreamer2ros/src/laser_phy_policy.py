import argparse
import collections
import functools
import os
import pathlib
import sys
import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
tf.get_logger().setLevel('DEBUG')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools

def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('.')
  config.seed = 0
  config.steps = 5e6
  config.eval_every = 1e4
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True
  config.precision = 32
  # Environment.
  config.task = 'dmc_walker_walk'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 1
  config.time_limit = 1000
  config.prefill = 5000
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  # Model.
  config.deter_size = 300
  config.stoch_size = 30
  config.num_units = 400
  config.phy_deter_size = 50
  config.phy_stoch_size = 5
  config.phy_num_units = 60
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 32
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 1000
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 100.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.9965
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  return config


class Dreamer(tools.Module):

  def __init__(self, config, actspace):#, writer):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._random = np.random.RandomState(config.seed)
    self._float = prec.global_policy().compute_dtype
    self._build_model()

  def __call__(self, obs, reset, state=None, training=True):
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)
    action, state = self.policy(obs, state, training)
    return action, state

  def policy(self, obs, env_state, phy_state, training):
    if env_state is None:
      env_latent = self._env_dynamics.initial(len(obs['laser']))
      action = tf.zeros((len(obs['laser']), self._actdim), self._float)
    else:
      env_latent, action = env_state
    if phy_state is None:
      phy_latent = self._phy_dynamics.initial(len(obs['laser']))
      action = tf.zeros((len(obs['laser']), self._actdim), self._float)
    else:
      phy_latent, action = phy_state
    obs = preprocess(obs, self._c)
    embed = self._encode(obs)
    env_latent, _ = self._env_dynamics.obs_step(env_latent, obs['prev_phy'], embed, sample=False)
    phy_latent, _ = self._phy_dynamics.obs_step(phy_latent, action, obs['input_phy'], sample=False)
    feat = tf.concat([self._env_dynamics.get_feat(env_latent), self._phy_dynamics.get_feat(phy_latent)],-1)
    if training:
      action = self._actor(feat).sample()
      #action = self._exploration(action, training)
    else:
      action = self._actor(feat).mode()
    env_state = (env_latent, action)
    phy_state = (phy_latent, action)
    return action, env_state, phy_state

  def load(self, path):
    print(path)
    self._phy_dynamics.load(os.path.join(path,'phy_dynamics_weights.pkl'))
    self._env_dynamics.load(os.path.join(path,'env_dynamics_weights.pkl'))
    self._actor.load(os.path.join(path,'actor_weights.pkl'))
    self._encode.load(os.path.join(path,'encoder_weights.pkl'))

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.LaserConvEncoder(self._c.cnn_depth, cnn_act)
    self._env_dynamics = models.RSSMv2(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._phy_dynamics = models.RSSMv2(self._c.phy_stoch_size, self._c.phy_deter_size, self._c.phy_deter_size)
    self._actor = models.ActionDecoder(self._actdim, 4, self._c.num_units, self._c.action_dist, init_std=self._c.action_init_std, act=act)
    obsr = {}
    obsr['laser'] = np.ones((1,256,1))
    obsr['physics'] = np.ones((3))
    obsr['input_phy'] = np.ones((1,3),dtype=np.float32)
    obsr['physics_d'] = np.ones((1,3),dtype=np.float32)
    obsr['prev_phy'] = np.ones((1,3),dtype=np.float32)
    obsr['image'] = np.ones((1,64,64,3))
    obsr['reward'] = np.ones((1,1))
    print('model built ok')
    embed = self._encode(preprocess(obsr, self._c))
    print('encoder initialized')
    env_latent = self._env_dynamics.initial(1)
    phy_latent = self._phy_dynamics.initial(1)
    action = tf.zeros((1, self._actdim), self._float)
    env_latent, _ = self._env_dynamics.obs_step(env_latent, obsr['prev_phy'], embed)
    phy_latent, _ = self._phy_dynamics.obs_step(phy_latent, action, obsr['input_phy'])
    feat = tf.concat([self._env_dynamics.get_feat(env_latent),self._phy_dynamics.get_feat(phy_latent)],-1)
    print('dynamics initialized')
    action = self._actor(feat).mode()
    print('actor initialized')

def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['input_phy'] = tf.cast(obs['physics'],dtype)
    obs['prev_phy'] = tf.cast(obs['physics_d'],dtype)
    obs['laser'] = tf.cast(1/obs['laser'] - 0.5, dtype)
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
  return obs

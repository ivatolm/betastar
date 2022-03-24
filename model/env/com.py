from multiprocessing import shared_memory
import numpy as np
import time

from configs.com_cfg import *


class ComServer:
  def __init__(self, cv_server, cv_client, sm_action_data, sm_state_data, sm_reward_data, sm_done_data):
    self.cv_server = cv_server
    self.cv_client = cv_client
    self.sm_action = shared_memory.SharedMemory(name=sm_action_data['name'])
    self.sm_state = shared_memory.SharedMemory(name=sm_state_data['name'])
    self.sm_reward = shared_memory.SharedMemory(name=sm_reward_data['name'])
    self.sm_done = shared_memory.SharedMemory(name=sm_done_data['name'])
    self.buff_action = np.ndarray(sm_action_data['shape'], dtype=sm_action_data['dtype'], buffer=self.sm_action.buf)
    self.buff_state = np.ndarray(sm_state_data['shape'], dtype=sm_state_data['dtype'], buffer=self.sm_state.buf)
    self.buff_reward = np.ndarray(sm_reward_data['shape'], dtype=sm_reward_data['dtype'], buffer=self.sm_reward.buf)
    self.buff_done = np.ndarray(sm_done_data['shape'], dtype=sm_done_data['dtype'], buffer=self.sm_done.buf)


  def send(self, action):
    with self.cv_server:
      self.buff_action[:] = action[:]

      with self.cv_client:
        self.cv_client.notify()
      
      self.cv_server.wait()


  def get(self):
    with self.cv_server:
      data = (self.buff_state[:],
              self.buff_reward[:],
              self.buff_done[:])

    return data


  def close(self):
    self.sm_action.close()
    self.sm_state.close()
    self.sm_reward.close()
    self.sm_done.close()


class ComClient:
  def __init__(self, cv_server, cv_client, sm_action_data, sm_state_data, sm_reward_data, sm_done_data):
    self.cv_server = cv_server
    self.cv_client = cv_client
    self.sm_action = shared_memory.SharedMemory(name=sm_action_data['name'])
    self.sm_state = shared_memory.SharedMemory(name=sm_state_data['name'])
    self.sm_reward = shared_memory.SharedMemory(name=sm_reward_data['name'])
    self.sm_done = shared_memory.SharedMemory(name=sm_done_data['name'])
    self.buff_action = np.ndarray(sm_action_data['shape'], dtype=sm_action_data['dtype'], buffer=self.sm_action.buf)
    self.buff_state = np.ndarray(sm_state_data['shape'], dtype=sm_state_data['dtype'], buffer=self.sm_state.buf)
    self.buff_reward = np.ndarray(sm_reward_data['shape'], dtype=sm_reward_data['dtype'], buffer=self.sm_reward.buf)
    self.buff_done = np.ndarray(sm_done_data['shape'], dtype=sm_done_data['dtype'], buffer=self.sm_done.buf)


  def send(self, state, reward, done):
    with self.cv_client:
      self.buff_state[:] = state[:]
      self.buff_reward[:] = reward[:]
      self.buff_done[:] = done[:]

      with self.cv_server:
        self.cv_server.notify()
      
      self.cv_client.wait()


  def get(self):
    with self.cv_client:
      data = self.buff_action[:]

    return data


  def close(self):
    self.sm_action.close()
    self.sm_state.close()
    self.sm_reward.close()
    self.sm_done.close()

import torch
import numpy as np

from configs.torch_cfg import *


class Agent:
	def __init__(self, env, memory):
		self.env = env
		self.memory = memory
		self._reset()


	def _reset(self):
		self.state = self.env.reset()
		self.total_reward = 0


	def play_step(self, net, epsilon=0):
		done_reward = None

		if np.random.random() < epsilon:
			action = np.random.randint(0, net.output_shape[0])
		else:
			state_a = np.array([self.state], copy=False)
			state_t = torch.tensor(state_a).to(DEVICE)
			prediction = net(state_t)
			_, act_v = torch.max(prediction, dim=1)
			action = int(act_v.item())
		
		next_state, reward, done, _ = self.env.step(action)
		self.total_reward += reward

		self.memory.push(self.state, action, reward, done, next_state)
		self.state = next_state

		if done:
			done_reward = self.total_reward
			self._reset()
		
		return done_reward

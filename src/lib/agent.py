import torch
import numpy as np

from configs.torch_cfg import *


class Agent:
	def __init__(self, env, memory):
		self.env = env
		self.memory = memory


	def reset(self):
		self.state = self.env.reset()
		self.total_reward = 0


	def play_step(self, net, epsilon):
		done_reward = None

		if np.random.random() < epsilon:
			action = np.random.randint(0, net.output_shape[0])
		else:
			state_t = torch.tensor(self.state).to(DEVICE)
			action = int(torch.max(net(state_t), dim=1)[1].item())

		next_state, reward, done, _ = self.env.step(action)
		self.memory.push(np.copy(self.state), np.copy(action), np.copy(reward), np.copy(done), np.copy(next_state))
		self.state = next_state

		self.total_reward += reward

		if done:
			done_reward = self.total_reward
		
		return done_reward

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
			actions_qs, actions = [], []
			for shape in net.output_shapes:
				actions_qs.append(None)
				actions.append(np.random.randint(0, shape[0]))
		else:
			state_t = torch.tensor(self.state).to(DEVICE)
			pred = net(state_t)
			actions_qs, actions = [], []
			for qs in pred:
				m = torch.max(qs, dim=1)
				actions_qs.append(m[0].item())
				actions.append(m[1].item())

		action = tuple(actions)

		next_state, reward, done, _ = self.env.step(action)
		self.memory.push(np.copy(self.state), np.copy(action), np.copy(reward), np.copy(done), np.copy(next_state))
		self.state = next_state

		self.total_reward += reward

		if done:
			done_reward = self.total_reward
		
		return done_reward, (self.state, actions, actions_qs)

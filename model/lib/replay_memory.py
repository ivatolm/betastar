import numpy as np
from collections import deque


class ReplayMemory:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)


	def push(self, *args):
		self.buffer.append(args)


	def sample(self, batch_size):
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)
		states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
		return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)


	def __len__(self):
		return len(self.buffer)

import numpy as np
from collections import deque


class ReplayMemory:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)


	def push(self, *args):
		self.buffer.append(args)


	def sample(self, batch_size, steps):
		indices = np.random.choice(len(self.buffer) - steps + 1, batch_size, replace=False)
		states, actions = zip(*[self.buffer[idx][:2] for idx in indices])
		dones = [self.buffer[idx][3] for idx in indices]
		next_states = [self.buffer[idx][4] for idx in indices]
		rewards = []
		for idx in indices:
			rewards.append([])
			for i in range(steps):
				index = idx + i
				if index < len(self.buffer):
					reward, done = self.buffer[index][2:4]
					rewards[-1].append(reward if not done else 0)
		return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)


	def __len__(self):
		return len(self.buffer)

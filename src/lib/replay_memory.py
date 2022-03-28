import numpy as np
from collections import deque



class ReplayMemory:
	def __init__(self, capacity):
		self.buffer = deque(maxlen=capacity)


	def push(self, *args):
		self.buffer.append(args)


	def sample(self, batch_size, steps):
		indices = np.random.choice(len(self.buffer), batch_size, replace=False)

		states, actions, rewards, dones, next_states = [], [], [], [], []
		for idx in indices:
			state, action, _, done, _ = self.buffer[idx]

			states.append(state)
			actions.append(action)
			curr_rewards = []

			done_index = None
			for i in range(steps):
				index = idx + i
				if index >= len(self.buffer):
					continue

				_, _, reward, _, _ = self.buffer[index]
				curr_rewards.append(reward)

				done_index = index
				if done:
					break

			if done_index is not None:
				_, _, _, done, next_state = self.buffer[done_index]
				dones.append(done)
				next_states.append(next_state)

			rewards.append(np.array(curr_rewards))

		return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(next_states)


	def sample_old(self, batch_size, steps):
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

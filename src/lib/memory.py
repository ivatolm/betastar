import collections as clct
import numpy as np

from ..types import Memory_t


class OneStepMemory(Memory_t):
  def __init__(self, capacity: int) -> None:
    self.capacity = capacity
    self.memory = clct.deque(maxlen=capacity)


  def push(self, state: np.array, action: tuple[int], reward: float, done: bool, next_state: np.array) -> None:
    self.memory.append(tuple([state, action, reward, done, next_state]))


  def sample(self, batch_size: int) -> tuple[np.array]:
    indices = np.random.choice(len(self.memory), batch_size, replace=False)
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for i in indices:
      state, action, reward, done, next_state = self.memory[i]
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      dones.append(done)
      next_states.append(next_state)
    return tuple([np.array(states), np.array(actions), np.array(rewards),
                  np.array(dones), np.array(next_states)])


  def size(self) -> int:
    return len(self.memory)


class NStepMemory(Memory_t):
  def __init__(self, capacity: int) -> None:
    self.capacity = capacity
    self.memory = clct.deque(maxlen=capacity)


  def push(self, state: np.array, action: tuple[int], reward: float, done: bool, next_state: np.array) -> None:
    self.memory.append(tuple([state, action, reward, done, next_state]))


  def sample(self, batch_size: int, steps: int, gamma: float) -> tuple[np.array]:
    indices = np.random.choice(self.size(), batch_size, replace=False)
    states, actions, rewards, next_gammas, dones, next_states = [], [], [], [], [], []

    for i in indices:
      state, action, _, _, _ = self.memory[i]
      states.append(state)
      actions.append(action)

      discounted_total_episode_reward = 0
      gamma_ = gamma
      for j in range(steps):
        k = i + j
        if k >= self.size():
          break

        _, _, reward, done, _ = self.memory[k]

        discounted_total_episode_reward += reward * gamma_
        gamma_ *= gamma

        if done:
          break

      rewards.append(discounted_total_episode_reward)
      next_gammas.append(gamma_)

      if k >= len(self.memory):
        k -= 1

      _, _, _, done, next_state = self.memory[k]
      dones.append(done)
      next_states.append(next_state)

    return tuple([np.array(states), np.array(actions), np.array(rewards),
                  np.array(next_gammas), np.array(dones), np.array(next_states)])


  def size(self) -> int:
    return len(self.memory)

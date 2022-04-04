import torch
import numpy as np

from ..types import Policy_t


class EpsilonGreedyPolicy(Policy_t):
  def __init__(self, epsilon_start: float, epsilon_end: float, epsilon_decay_length: float) -> None:
    self.epsilon_start = epsilon_start
    self.epsilon_end = epsilon_end
    self.epsilon_decay_length = epsilon_decay_length

    self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_length

    self.epsilon = self.epsilon_start


  def get_action(self, qs: tuple[float]) -> int:
    if np.random.random() <= self.epsilon:
      action = np.random.randint(0, len(qs))
    else:
      action = int(torch.argmax(qs))

    return action


  def get_pure_action(self, qs: tuple[float]) -> int:
    action = np.argmax(qs)
    return action


  def update(self) -> None:
    self.epsilon = np.maximum(self.epsilon - self.epsilon_decay, self.epsilon_end)


  def get_epsilon(self) -> float:
    return self.epsilon


class NdimEpsilonGreedyPolicy(Policy_t):
  def __init__(self, epsilon_start: float, epsilon_end: float, epsilon_decay_length: float) -> None:
    self.epsilon_start = epsilon_start
    self.epsilon_end = epsilon_end
    self.epsilon_decay_length = epsilon_decay_length

    self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_length

    self.epsilon = self.epsilon_start


  def get_action(self, qs_l: tuple[torch.tensor]) -> tuple[int]:
    if np.random.random() <= self.epsilon:
      action = tuple([np.random.randint(0, qs.size(dim=1)) for qs in qs_l])
    else:
      action = tuple([int(torch.argmax(qs)) for qs in qs_l])

    return action


  def get_pure_action(self, qs_l: tuple[np.array]) -> tuple[int]:
    action = tuple([np.argmax(qs) for qs in qs_l])
    return action


  def update(self) -> None:
    self.epsilon = np.maximum(self.epsilon - self.epsilon_decay, self.epsilon_end)


  def get_epsilon(self) -> float:
    return self.epsilon

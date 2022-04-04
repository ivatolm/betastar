import torch
import torch.nn as nn
import numpy as np

from ..types import Loss_t, Model_t
from configs.torch_cfg import DEVICE


class OneStepLoss(Loss_t):
  def __init__(self, gamma: float, loss_func_name: str) -> None:
    self.gamma = gamma
    self.loss_func = None
    match loss_func_name:
      case "mse":
        self.loss_func = nn.MSELoss()
      case "huber":
        self.loss_func = nn.HuberLoss()


  def get_loss(self, batch: np.array, model: Model_t, target_model: Model_t) -> torch.tensor:
    states, actions, rewards, dones, next_states = batch
    states_t      = torch.as_tensor(states).to(DEVICE)
    actions_t     = torch.as_tensor(actions).to(DEVICE)
    rewards_t     = torch.as_tensor(rewards).to(DEVICE)
    dones_t       = torch.as_tensor(dones).to(DEVICE)
    next_states_t = torch.as_tensor(next_states).to(DEVICE)

    model_qs        = model(states_t)
    target_model_qs = target_model(next_states_t)

    qs_t = model_qs.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

    next_qs_t          = target_model_qs.max(1)[0]
    next_qs_t[dones_t] = 0
    next_qs_t          = next_qs_t.detach()
    
    expected_qs_t = next_qs_t * self.gamma + rewards_t

    return self.loss_func(qs_t, expected_qs_t)


class NStepNdimLoss(Loss_t):
  def __init__(self, loss_func_name: str) -> None:
    self.loss_func = None
    match loss_func_name:
      case "mse":
        self.loss_func = nn.MSELoss()
      case "huber":
        self.loss_func = nn.HuberLoss()


  def get_loss(self, batch: tuple, model: Model_t, target_model: Model_t) -> torch.tensor:
    states, actions, rewards, next_gammas, dones, next_states = batch

    states_t      = torch.as_tensor(states).to(DEVICE)
    actions_t     = torch.as_tensor(actions).to(DEVICE)
    rewards_t     = torch.as_tensor(rewards).to(DEVICE)
    next_gammas_t = torch.as_tensor(next_gammas).to(DEVICE)
    dones_t       = torch.as_tensor(dones).to(DEVICE)
    next_states_t = torch.as_tensor(next_states).to(DEVICE)

    model_qs        = model(states_t)
    target_model_qs = target_model(next_states_t)

    dims_num    = len(model_qs)
    actions_num = len(actions_t)

    losses = torch.zeros(dims_num).to(DEVICE)
    for i in range(dims_num):
      indices_t = torch.full(tuple([actions_num]), i).to(DEVICE)
      indices_t = indices_t.unsqueeze(-1)

      dim_qs_t = model_qs[i]
      qs_t     = dim_qs_t.gather(1, actions_t.gather(1, indices_t)).squeeze(-1)

      next_dim_qs_t      = target_model_qs[i]
      next_dim_qs_t      = next_dim_qs_t.detach()
      next_qs_t          = next_dim_qs_t.max(1)[0]
      next_qs_t[dones_t] = 0

      expected_qs_t = rewards_t + next_qs_t * next_gammas_t

      losses[i] = self.loss_func(qs_t, expected_qs_t)

    return losses.mean()

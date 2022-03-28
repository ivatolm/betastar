import torch
import torch.nn as nn
import numpy as np
import gc

from configs.torch_cfg import *


def loss_mse_old(batch, net, target_net, gamma):
  states, actions, rewards, dones, next_states = batch

  states_t = torch.tensor(states).to(DEVICE)
  actions_t = torch.tensor(actions).to(DEVICE)
  rewards_t = torch.tensor(rewards).to(DEVICE)
  next_states_t = torch.tensor(next_states).to(DEVICE)
  done_mask = torch.tensor(dones).to(DEVICE)

  state_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

  steps = rewards.shape[1]
  gammas_t = torch.tensor([gamma ** i for i in range(steps)]).to(DEVICE)

  rewards_sums = (rewards_t * gammas_t).sum(1)
  rewards_sums = rewards_sums.detach()
  next_state_values = target_net(next_states_t).max(1)[0]
  next_state_values[done_mask] = 0.0
  next_state_values = next_state_values.detach()

  expected_state_values = rewards_sums + next_state_values

  return nn.MSELoss()(state_values.float(), expected_state_values.float())


def loss_mse(batch, net, target_net, gamma):
  states, actions, rewards, dones, next_states = batch

  states_t      = torch.tensor(np.array(states)).to(DEVICE)
  actions_t     = torch.tensor(np.array(actions)).to(DEVICE)
  rewards_o     = rewards
  next_states_t = torch.tensor(np.array(next_states)).to(DEVICE)
  done_mask     = torch.tensor(np.array(dones)).to(DEVICE)

  state_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

  next_state_values = target_net(next_states_t).max(1)[0]
  next_state_values[done_mask] = 0.0
  next_state_values = next_state_values.detach()

  exp_state_values = []
  for i, rewards_l in enumerate(rewards_o):
    gammas_l = [gamma ** i for i in range(len(rewards_l))]

    rewards_a = torch.tensor(rewards_l).to(DEVICE)
    gammas_a = torch.tensor(gammas_l).to(DEVICE)

    exp_state_value = (rewards_a * gammas_a).sum() + next_state_values[i] * gamma**len(rewards_l)
    exp_state_values.append(exp_state_value)

  exp_state_values_t = torch.tensor(exp_state_values).to(DEVICE)

  return nn.MSELoss()(state_values.float(), exp_state_values_t.float())

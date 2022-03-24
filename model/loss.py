import torch
import torch.nn as nn

from configs.torch_cfg import *


def loss_mse(batch, net, target_net, gamma):
  states, actions, rewards, dones, next_states = batch

  states_t = torch.tensor(states).to(DEVICE)
  actions_t = torch.tensor(actions).to(DEVICE)
  rewards_t = torch.tensor(rewards).to(DEVICE)
  next_states_t = torch.tensor(next_states).to(DEVICE)
  done_mask = torch.tensor(dones).to(DEVICE)

  state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
  next_state_values = target_net(next_states_t).max(1)[0]
  next_state_values[done_mask] = 0.0
  next_state_values = next_state_values.detach()

  expected_state_action_values = next_state_values * gamma + rewards_t
  return nn.MSELoss()(state_action_values.float(), expected_state_action_values.float())

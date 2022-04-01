import torch
import torch.nn as nn
import numpy as np
import logging

from configs.torch_cfg import *


def _diff(batch, net, target_net, gamma):
  states, actions, rewards, dones, next_states = batch

  states_t      = torch.tensor(np.array(states)).to(DEVICE)
  actions_t     = torch.tensor(np.array(actions)).to(DEVICE)
  rewards_o     = rewards
  next_states_t = torch.tensor(np.array(next_states)).to(DEVICE)
  done_mask     = torch.tensor(np.array(dones)).to(DEVICE)

  res = []
  pred = net(states_t)
  target_pred = target_net(next_states_t)
  for dim_id in range(len(pred)):
    indices = torch.full((len(actions_t),), dim_id).to(DEVICE)
    qs = pred[dim_id].gather(1, actions_t.gather(1, indices.unsqueeze(-1))).squeeze(-1)

    next_qs = torch.max(target_pred[dim_id], dim=1)[0].detach()
    next_qs[done_mask] = 0

    gammas_t = torch.tensor([gamma**i for i in range(len(max(rewards_o, key=len)) + 1)]).to(DEVICE)

    exp_qs_t = None
    for i, rewards_l in enumerate(rewards_o):
      rewards_a = torch.tensor(rewards_l).to(DEVICE)

      exp_q = (rewards_a * gammas_t[:len(rewards_l)]).sum() + next_qs[i] * gammas_t[len(rewards_l)]
      exp_q = exp_q.unsqueeze(-1)
      if exp_qs_t is None:
        exp_qs_t = exp_q
      else:
        exp_qs_t = torch.cat((exp_qs_t, exp_q))
    res.append((qs.float(), exp_qs_t.float()))

  return res


def loss(batch, net, target_net, gamma, func_name):
  func = None
  match func_name:
    case "mse":
      func = nn.MSELoss()
    case "huber":
      func = nn.HuberLoss()

  if func is None:
    logging.error("loss: unknown loss function, falling to mse")
    func = nn.MSELoss()

  diff = _diff(batch, net, target_net, gamma)

  losses = torch.zeros(len(diff)).to(DEVICE)
  for i, dim_diff in enumerate(diff):
    losses[i] = func(*dim_diff)

  return losses.mean()

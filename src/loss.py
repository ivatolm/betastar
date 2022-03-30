from turtle import done
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

  dim_qs = []
  pred = net(states_t)
  for dim_id in range(len(pred)):
    indices = torch.full((len(actions_t),), dim_id).to(DEVICE).unsqueeze(-1)
    qs = pred[dim_id].gather(1, actions_t.gather(1, indices)).squeeze(-1)
    dim_qs.append(qs)
  
  next_dim_qs = []
  target_pred = target_net(next_states_t)
  for dim_id in range(len(target_pred)):
    m = torch.max(target_pred[dim_id], dim=1)[0].detach()
    m[done_mask] = 0
    next_dim_qs.append(m)

  res = []
  for dim_id in range(len(next_dim_qs)):
    exp_qs = []
    for i, rewards_l in enumerate(rewards_o):
      gammas_l = [gamma ** i for i in range(len(rewards_l))]

      rewards_a = torch.tensor(rewards_l).to(DEVICE)
      gammas_a = torch.tensor(gammas_l).to(DEVICE)

      exp_q = (rewards_a * gammas_a).sum() + next_dim_qs[dim_id][i] * gamma**len(rewards_l)
      exp_qs.append(exp_q)
    
    exp_qs_t = torch.tensor(exp_qs).to(DEVICE)
    res.append((dim_qs[dim_id].float(), exp_qs_t.float()))

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

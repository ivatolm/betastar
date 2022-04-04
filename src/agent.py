import torch.optim as optim
import numpy as np
import copy
import time

from .types import *
from configs.torch_cfg import DEVICE


class DDQNAgent:
  def __init__(self, model: Model_t, memory: Memory_t, loss: Loss_t, policy: Policy_t, optimizer: optim, conf: dict) -> None:
    self.model        = model.to(DEVICE)
    self.target_model = copy.deepcopy(self.model).to(DEVICE)
    self.memory       = memory
    self.loss         = loss
    self.policy       = policy
    self.optimizer    = optimizer
    self.conf         = conf
    self.iter_cntr    = 0

    self.episode_stats = {"rewards": [],
                          "losses": [],
                          "qs": [],
                          "frame_times": []}
    self.stats         = {"iteration": 0,
                          "total_reward": 0,
                          "mean_loss": 0,
                          "mean_qs": 0,
                          "mean_frame_time": 0,
                          "epsilon": 0}

    self.env                   = None
    self.is_env_done_flag      = True


  def iter(self) -> None:
    frame_start = time.time()

    qs = self.model(torch.as_tensor(self.state))
    action = self.policy.get_action(qs)
    next_state, reward, done, _ = self.env.step(action)
    self.memory.push(self.state, action, reward, done, next_state)
    self.state = next_state
    if done:
      self.stats["iteration"]       = self.iter_cntr
      self.stats["total_reward"]    = sum(self.episode_stats["rewards"])
      self.stats["mean_loss"]       = np.mean(self.episode_stats["losses"])
      self.stats["mean_qs"]         = np.mean(self.episode_stats["qs"])
      self.stats["mean_frame_time"] = np.mean(self.episode_stats["frame_times"])
      self.stats["epsilon"]         = self.policy.get_epsilon()
      self.episode_stats            = {"rewards": [],
                                       "losses": [],
                                       "qs": [],
                                       "frame_times": []}
      self.is_env_done_flag         = True

    self.policy.update()

    if self.memory.size() >= self.conf["batch_size"]:
      if self.iter_cntr % self.conf["merge_freq"] == 0:
        self.target_model.load_state_dict(self.model.state_dict())

      batch = self.memory.sample(self.conf["batch_size"],
                                 self.conf["steps"],
                                 self.conf["gamma"])
      loss = self.loss.get_loss(batch, self.model, self.target_model)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      frame_time = time.time() - frame_start

      self.episode_stats["rewards"].append(reward)
      self.episode_stats["losses"].append(loss.item())
      self.episode_stats["qs"].append(qs[0].mean().item())
      self.episode_stats["frame_times"].append(frame_time)

    self.iter_cntr += 1


  def set_env(self, env) -> None:
    self.env              = env
    self.is_env_done_flag = False
    self.state            = self.env.reset()


  def remove_env(self) -> None:
    self.env   = None
    self.state = None


  def is_env_done(self) -> bool:
    return self.is_env_done_flag


  def get_stats(self) -> dict:
    return self.stats


  def get_model(self) -> Model_t:
    return self.model
  

  def get_memory(self) -> Memory_t:
    return self.memory

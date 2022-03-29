from src.env.env import Env
from src.lib.replay_memory import ReplayMemory
from src.lib.agent import Agent
from src.lib.dqn import DQN
from src.env.bots.bot_01 import Bot
from src.loss import loss_mse
from src.tools.measurer import Measurer
from src.utils import train_cycle, gen_plan_str
from src.graphics import Graphics

from collections import deque
from pathlib import Path
import torch.optim as optim
import numpy as np
import pickle
import copy
import os
import logging

from configs.torch_cfg import *
from configs.data_cfg import *


def train_pipeline(plan, load_version=None, save_version=VERSION, graphics=None):
  base_plan = plan["BASE"]
  logging.info(f"train_pipeline: {gen_plan_str('BASE', base_plan)}")
  if load_version is not None:
    logging.info(f"train_pipeline: loading version {load_version}")
    net = torch.load(SAVES_DIR + '/' + load_version + ".net")
    with open(SAVES_DIR + '/' + load_version + ".mem", "rb") as file:
      memory = pickle.load(file)
    with open(SAVES_DIR + '/' + load_version + ".ost", "rb") as file:
      optimizer_state = pickle.load(file)
  else:
    logging.info("train_pipeline: creating network, memory")
    net = DQN(base_plan["dqn_input_shape"], base_plan["dqn_output_shape"]).to(DEVICE)
    memory = ReplayMemory(base_plan["memory_capacity"])
    optimizer_state = optim.Adam(net.parameters()).state_dict()

  logging.info(net)

  if graphics is not None:
    graphics = Graphics()

  for plan_name, env_plan in plan.items():
    if plan_name == "BASE":
      continue
    logging.info(f"train_pipeline: training by '{plan_name}' config")
    logging.info(f"train_pipeline: {gen_plan_str(plan_name, env_plan)}")
    net, memory, optimizer_state = env_train_pipeline(base_plan, env_plan,
                                                      net, memory, optimizer_state,
                                                      metrics_version=save_version + '/' + plan_name,
                                                      graphics=graphics)

  logging.info("train_pipeline: training finished")

  logging.info(f"train_pipeline: saving to version {save_version}")
  if not Path(SAVES_DIR).exists():
    os.mkdir(SAVES_DIR)
  torch.save(net, SAVES_DIR + '/' + save_version + ".net")
  with open(SAVES_DIR + '/' + save_version + ".mem", "wb") as file:
    pickle.dump(memory, file)
  with open(SAVES_DIR + '/' + save_version + ".ost", "wb") as file:
    pickle.dump(optimizer_state, file)


def env_train_pipeline(base_plan, env_plan, net, memory, optimizer_state, metrics_version, graphics=None):
  target_net = copy.deepcopy(net)

  optimizer = optim.Adam(net.parameters(), lr=env_plan["learning_rate"])
  optimizer.load_state_dict(optimizer_state)
  for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
  env = Env(base_plan, env_plan["env_map"], (Bot, (base_plan["env_view_size"], base_plan["env_map_size"])))
  epsilon = env_plan["epsilon_max"]

  agent = Agent(env, memory)
  measurer = Measurer(env_plan["env_map"], metrics_version)

  rewards = deque(maxlen=100)
  for episode in range(env_plan["episodes_num"]):
    loss = None
    agent.reset()
    reward = None
    while reward is None:
      reward, info = agent.play_step(net, epsilon)

      if graphics is not None:
        graphics.update(*info)

      if len(memory) >= env_plan["min_memory_capacity"]:
        if len(memory) == env_plan["min_memory_capacity"]:
          logging.info("train: training started")

        loss = train_cycle(net, target_net, memory, optimizer, loss_mse, env_plan["batch_size"], env_plan["steps"], env_plan["gamma"])

    if len(memory) >= env_plan["min_memory_capacity"]:
      epsilon = np.maximum(epsilon * env_plan["epsilon_decay"], env_plan["epsilon_min"])

    if episode % env_plan["merge_freq"] == 0:
      target_net.load_state_dict(net.state_dict())
      logging.info("train: models syncronized")

    rewards.append(reward)

    logging.info(f"train: "
                 f"episode {episode + 1}/{env_plan['episodes_num']}, "
                 f"mean reward {round(np.mean(rewards), 3)}, "
                 f"epsilon {round(epsilon, 3)}")
    measurer.add_value("mean_reward", np.mean(rewards), episode)
    measurer.add_value("epsilon", epsilon, episode)
    logging.info(loss)

  del env

  return net, memory, optimizer.state_dict()

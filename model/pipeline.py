from model.env.env import Env
from model.lib.replay_memory import ReplayMemory
from model.lib.agent import Agent
from model.lib.dqn import DQN
from model.env.bots.bot_01 import Bot
from model.loss import loss_mse
from model.tools.measurer import Measurer
from model.utils import train_cycle

from collections import deque
import torch.optim as optim
import numpy as np
import time
import copy
import logging

from configs.torch_cfg import *
from configs.pipeline_cfg import *


def pipeline():
  logging.info("pipeline: creating network, memory")
  net = DQN(DQN_INPUT_SHAPE, DQN_OUTPUT_SHAPE).to(DEVICE)
  memory = ReplayMemory(MEMORY_CAPACITY)
  logging.info(net)

  logging.info("pipeline: starting to train in 'env_0'")
  pipeline_env_0(10, net, memory)

  logging.info("pipeline: training finished")


def pipeline_env_0(episodes, net, memory):
  target_net = copy.deepcopy(net)

  optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
  env = Env(ENV_MAP, (Bot, (ENV_VIEW_SIZE, ENV_MAP_SIZE)))
  epsilon = EPSILON_MAX

  agent = Agent(env, memory)
  measurer = Measurer("env_0")

  for episode in range(episodes):
    rewards = []

    agent.reset()
    reward = None
    while reward is None:
      reward = agent.play_step(net, epsilon)
      if len(memory) >= MIN_MEMORY_CAPACITY:
        if len(memory) == MIN_MEMORY_CAPACITY:
          logging.info("train: training started")

        train_cycle(net, target_net, memory, optimizer, loss_mse, GAMMA, BATCH_SIZE)

    if len(memory) >= MIN_MEMORY_CAPACITY:
      epsilon = np.maximum(epsilon * EPSILON_DECAY, EPSILON_MIN)

    if episode % MERGE_FREQ == 0:
      target_net.load_state_dict(net.state_dict())
      logging.info("train: models syncronized")

    rewards.append(reward)

    logging.info(f"train: "
                 f"episode {episode + 1}/{episodes}, "
                 f"mean reward {round(np.mean(rewards), 3)}, "
                 f"epsilon {round(epsilon, 3)}")
    measurer.add_value("mean_reward", np.mean(rewards), episode)
    measurer.add_value("epsilon", epsilon, episode)

  del env


def pipeline_old():
  env = Env(ENV_MAP, (Bot, (ENV_VIEW_SIZE, ENV_MAP_SIZE)))
  memory = ReplayMemory(MEMORY_CAPACITY)
  agent = Agent(env, memory)

  net = DQN(DQN_INPUT_SHAPE, DQN_OUTPUT_SHAPE).to(DEVICE)
  target_net = DQN(DQN_INPUT_SHAPE, DQN_OUTPUT_SHAPE).to(DEVICE)

  optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
  
  game_cntr = 0
  frame_cntr = 0

  frame_game_last = 0
  frame_timer = time.time()

  rewards_memory = deque([], maxlen=100)
  best_mean_reward = None

  epsilon = EPSILON_MAX
  while True:
    frame_cntr += 1

    reward = agent.play_step(net, epsilon)
    if reward is not None:
      game_cntr += 1

      if len(memory) >= MIN_MEMORY_CAPACITY:
        epsilon = np.maximum(epsilon * EPSILON_DECAY, EPSILON_MIN)

      if game_cntr % MERGE_FREQ == 0:
        target_net.load_state_dict(net.state_dict())

      rewards_memory.append(reward)
      mean_reward = np.mean(rewards_memory)

      now = time.time()
      speed = (frame_cntr - frame_game_last) / (now - frame_timer)
      frame_timer = now
      frame_game_last = frame_cntr

      pretty_mean_reward = round(mean_reward, 3)
      pretty_epsilon = round(epsilon, 3)
      pretty_speed = round(speed, 3)
      print(f"{frame_cntr}, {game_cntr}: mean {pretty_mean_reward}, eps {pretty_epsilon}, speed {pretty_speed}")

      if best_mean_reward is None or best_mean_reward < mean_reward:
        torch.save(net.state_dict(), "best.dat")
        if best_mean_reward is not None:
          print(f"Best mean reward {best_mean_reward} => {mean_reward}")
        best_mean_reward = mean_reward

    if len(memory) >= MIN_MEMORY_CAPACITY:
      optimizer.zero_grad()
      batch = memory.sample(BATCH_SIZE)
      loss_t = loss_mse(batch, net, target_net, GAMMA)
      loss_t.backward()
      optimizer.step()

from model.env.env import Env
from model.lib.replay_memory import ReplayMemory
from model.lib.agent import Agent
from model.lib.dqn import DQN
from model.env.bots.bot_01 import Bot

from collections import deque
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

from configs.torch_cfg import *
from configs.pipeline_cfg import *


def pipeline():
  env = Env(ENV_MAP, Bot(ENV_VIEW_SIZE, ENV_MAP_SIZE))
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
      def calc_loss(batch, net, target_net):
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

        expected_state_action_values = next_state_values * GAMMA + rewards_t
        return nn.MSELoss()(state_action_values.float(), expected_state_action_values.float())

      optimizer.zero_grad()
      batch = memory.sample(BATCH_SIZE)
      loss_t = calc_loss(batch, net, target_net)
      loss_t.backward()
      optimizer.step()

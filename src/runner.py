from .agent import DDQNAgent
from .lib.model import NdimDDQNModel, DQNModel
from .lib.loss import NStepNdimLoss, OneStepLoss
from .lib.policy import NdimEpsilonGreedyPolicy, EpsilonGreedyPolicy
from .lib.memory import NStepMemory, OneStepMemory

import gym
import pickle

def train():
  # model = DQNModel((4,), (2,))
  # loss = OneStepLoss(1.0, "mse")
  # policy = EpsilonGreedyPolicy(1, 0.1, 10000)
  # memory = OneStepMemory(100000)
  # env = gym.make("CartPole-v1")

  model = NdimDDQNModel((4,), ((2,),))
  loss = NStepNdimLoss("mse")
  policy = NdimEpsilonGreedyPolicy(1, 0.1, 5000)
  memory = NStepMemory(100000)
  env = gym.make("CartPole-v1")

  conf = {
    "learning_rate": 0.01,
    "merge_freq": 10,
    "batch_size": 1024,
    "steps": 2,
    "gamma": 0.99
  }

  # agent = DDQNAgent(model, loss, policy, memory, env, conf)
  with open("pickl", 'rb') as file:
    agent = pickle.load(file)
  for i in range(500 * 10):
    agent.iter()
    if i % 500 == 0:
      print(agent.get_stats())

  with open("pickl", 'wb') as file:
    pickle.dump(agent, file)


if __name__ == "__main__":
  train()

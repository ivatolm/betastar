from pathlib import Path
import torch.optim as optim
import pickle
import os
import copy
import logging

from .env.env import Env
from .measurer import Measurer

from .agent import DDQNAgent
from .lib.model import NdimDDQNModel
from .lib.loss import NStepNdimLoss
from .lib.policy import NdimEpsilonGreedyPolicy
from .lib.memory import NStepMemory

from configs.torch_cfg import *
from configs.data_cfg import *


def train_pipeline(plan, load_version=None, save_version=VERSION):
  base_plan = plan["BASE"]
  if load_version is not None:
    logging.info(f"train_pipeline: loading version {load_version}")
    with open(SAVES_DIR + '/' + load_version + ".model", "rb") as file:
      model = pickle.load(file)
    with open(SAVES_DIR + '/' + load_version + ".mem", "rb") as file:
      memory = pickle.load(file)
  else:
    logging.info("train_pipeline: creating model, memory")
    model = NdimDDQNModel(base_plan["model_input_shape"], base_plan["model_output_shapes"])
    memory = NStepMemory(base_plan["memory_capacity"])

  logging.info(model)

  for plan_name, env_plan in plan.items():
    if plan_name == "BASE":
      continue
    logging.info(f"train_pipeline: training by '{plan_name}' plan")
    model, memory = env_train_pipeline(base_plan, env_plan, model, memory,
                                       metrics_version=save_version)

  logging.info("train_pipeline: training finished")

  logging.info(f"train_pipeline: saving to version {save_version}")
  if not Path(SAVES_DIR).exists():
    os.mkdir(SAVES_DIR)
  with open(SAVES_DIR + '/' + save_version + ".model", "wb") as file:
    pickle.dump(model, file)
  with open(SAVES_DIR + '/' + save_version + ".mem", "wb") as file:
    pickle.dump(memory, file)


def env_train_pipeline(base_plan, env_plan, model, memory, metrics_version):
  loss = NStepNdimLoss(env_plan["loss_func"])
  policy = NdimEpsilonGreedyPolicy(env_plan["epsilon_start"], env_plan["epsilon_end"], env_plan["epsilon_decay_length"])
  optimizer = optim.Adam(model.parameters(), lr=env_plan["learning_rate"])
  agent = DDQNAgent(model, memory, loss, policy, optimizer, {"merge_freq": env_plan["merge_freq"],
                                                             "batch_size": env_plan["batch_size"],
                                                             "steps": env_plan["steps"],
                                                             "gamma": env_plan["gamma"]})

  measurer = Measurer(env_plan["env_map"], metrics_version)
  env = Env(base_plan, env_plan, (base_plan["env_view_size"], base_plan["env_map_size"]))

  agent.set_env(env)
  for i in range(env_plan["iterations"]):
    agent.iter()

    if agent.is_env_done():
      agent.remove_env()
      agent.set_env(env)
      stats = agent.get_stats()
      logging.info(f"train: "
                  f"iteration: {stats['iteration']}, "
                  f"total_reward: {round(stats['total_reward'], 3)}, "
                  f"mean_loss: {round(stats['mean_loss'], 3)}, "
                  f"mean_qs: {round(stats['mean_qs'], 3)}, "
                  f"mean_frame_time: {round(stats['mean_frame_time'], 3)}, "
                  f"epsilon: {round(stats['epsilon'], 3)}")
      measurer.add_value("total_reward", stats["total_reward"], stats["iteration"])
      measurer.add_value("mean_loss", stats["mean_loss"], stats["iteration"])
      measurer.add_value("mean_qs", stats["mean_qs"], stats["iteration"])
      measurer.add_value("mean_frame_time", stats["mean_frame_time"], stats["iteration"])
      measurer.add_value("epsilon", stats["epsilon"], stats["iteration"])
  agent.remove_env()

  measurer.close()

  return tuple([copy.deepcopy(agent.get_model()),
                copy.deepcopy(agent.get_memory())])

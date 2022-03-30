from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_multiple_games, a_run_multiple_games_nokill, GameMatch, Map
from sc2.data import Race, Difficulty
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Process, shared_memory, Condition
import numpy as np
import asyncio

from .com import ComServer


def run_multiple_games_no_kill(matches):
  return asyncio.get_event_loop().run_until_complete(a_run_multiple_games_nokill(matches))


class Env:
  def __init__(self, base_plan, env_plan, bot_data):
    self.base_plan = base_plan
    self.env_plan = env_plan
    self.bot_data = bot_data

    self.cv_server = Condition()
    self.cv_client = Condition()
    self.sm_action = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_action_shape"], dtype=np.float64).nbytes)
    self.sm_state = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_state_shape"], dtype=np.float64).nbytes)
    self.sm_reward = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_reward_shape"], dtype=np.float64).nbytes)
    self.sm_done = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_done_shape"], dtype=np.bool8).nbytes)

    self.com_args = (self.cv_server,
                     self.cv_client,
                     {"shape": self.base_plan["env_action_shape"], "dtype": np.float64, "name": self.sm_action.name},
                     {"shape": self.base_plan["env_state_shape"], "dtype": np.float64, "name": self.sm_state.name},
                     {"shape": self.base_plan["env_reward_shape"], "dtype": np.float64, "name": self.sm_reward.name},
                     {"shape": self.base_plan["env_done_shape"], "dtype": np.bool8, "name": self.sm_done.name})
    self.com = None

    matches = []
    for _ in range(env_plan["episodes_num"]):
      matches.append(GameMatch(maps.get(env_plan["env_map"]),
                               [Bot(Race.Terran, self.bot_data[0](self.bot_data[1], self.com_args)),
                                Computer(Race.Protoss, Difficulty.Medium)]))

    self.game_runner = Process(target=run_multiple_games_no_kill, args=(matches,))
    self.game_runner.start()


  def __del__(self):
    self.com.notify()
    self.game_runner.join()
    del self.com
    self.sm_action.close()
    self.sm_state.close()
    self.sm_reward.close()
    self.sm_done.close()
    self.sm_action.unlink()
    self.sm_state.unlink()
    self.sm_reward.unlink()
    self.sm_done.unlink()


  def reset(self):
    del self.com
    self.com = ComServer(*self.com_args)

    return np.zeros(shape=self.base_plan["env_state_shape"], dtype=np.float64)


  def step(self, action):
    self.com.send(action)
    self.state, self.reward, self.done = self.com.get()
    return self.state, self.reward[0], self.done[0], ""

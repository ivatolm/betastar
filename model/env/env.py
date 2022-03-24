from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Process, shared_memory, Condition
import numpy as np

from .com import ComServer

from configs.pipeline_cfg import *


class Env:
  def __init__(self, map_name, bot_data):
    self.map_name = map_name
    self.bot_data = bot_data

    self.cv_server = Condition()
    self.cv_client = Condition()
    self.sm_action = shared_memory.SharedMemory(create=True, size=np.zeros(shape=ENV_ACTION_SHAPE, dtype=np.float64).nbytes)
    self.sm_state = shared_memory.SharedMemory(create=True, size=np.zeros(shape=ENV_STATE_SHAPE, dtype=np.float64).nbytes)
    self.sm_reward = shared_memory.SharedMemory(create=True, size=np.zeros(shape=ENV_REWARD_SHAPE, dtype=np.float64).nbytes)
    self.sm_done = shared_memory.SharedMemory(create=True, size=np.zeros(shape=ENV_DONE_SHAPE, dtype=np.bool8).nbytes)

    self.com_args = (self.cv_server,
                     self.cv_client,
                     {"shape": ENV_ACTION_SHAPE, "dtype": np.float64, "name": self.sm_action.name},
                     {"shape": ENV_STATE_SHAPE, "dtype": np.float64, "name": self.sm_state.name},
                     {"shape": ENV_REWARD_SHAPE, "dtype": np.float64, "name": self.sm_reward.name},
                     {"shape": ENV_DONE_SHAPE, "dtype": np.bool8, "name": self.sm_done.name})
    self.com = ComServer(*self.com_args)

    self.game = None


  def __del__(self):
    if self.game is not None:
      self.com.notify()
      self.game.join()
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
    def _game_process(self):
      bot_class = self.bot_data[0]
      ai_args = self.bot_data[1]
      bot_args = (ai_args, self.com_args)
      run_game(maps.get(self.map_name), [
        Bot(Race.Zerg, bot_class(*bot_args)),
        Computer(Race.Protoss, Difficulty.Medium)
      ], realtime=False)
    
    if self.game is not None:
      self.com.notify()
      self.game.join()

    del self.com
    self.com = ComServer(*self.com_args)

    self.game = Process(target=_game_process, args=(self,))
    self.game.start()

    return np.zeros(shape=ENV_STATE_SHAPE, dtype=np.float64)


  def step(self, action):
    self.com.send([action])
    self.state, self.reward, self.done = self.com.get()
    return self.state, self.reward[0], self.done[0], ""

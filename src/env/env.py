from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import GameMatch, run_game
from sc2.data import Race, Difficulty
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Process, shared_memory, Condition, Value
import numpy as np

from .com import ComServer


class Env:
  def __init__(self, base_plan: dict, env_plan: dict, bot_data: tuple, graphics=None) -> None:
    self.base_plan = base_plan
    self.env_plan = env_plan
    self.bot_data = bot_data
    self.graphics = graphics

    self.cv_server = Condition()
    self.cv_client = Condition()
    self.die_flag  = Value('i', 0)
    self.sm_action = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_action_shape"], dtype=np.float64).nbytes)
    self.sm_state  = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_state_shape"], dtype=np.float64).nbytes)
    self.sm_reward = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_reward_shape"], dtype=np.float64).nbytes)
    self.sm_done   = shared_memory.SharedMemory(create=True, size=np.zeros(shape=self.base_plan["env_done_shape"], dtype=np.bool8).nbytes)

    self.com_args = (self.cv_server,
                     self.cv_client,
                     self.die_flag,
                     {"shape": self.base_plan["env_action_shape"], "dtype": np.float64, "name": self.sm_action.name},
                     {"shape": self.base_plan["env_state_shape"], "dtype": np.float64, "name": self.sm_state.name},
                     {"shape": self.base_plan["env_reward_shape"], "dtype": np.float64, "name": self.sm_reward.name},
                     {"shape": self.base_plan["env_done_shape"], "dtype": np.bool8, "name": self.sm_done.name})

    self.com = ComServer(*self.com_args)
    self.game = None


  def stop(self) -> None:
    if self.game is not None:
      self.com.notify()
      self.game.join()
    self.sm_action.close()
    self.sm_state.close()
    self.sm_reward.close()
    self.sm_done.close()
    self.sm_action.unlink()
    self.sm_state.unlink()
    self.sm_reward.unlink()
    self.sm_done.unlink()


  def reset(self) -> np.array:
    def _game_process(self):
      run_game(maps.get(self.env_plan["env_map"]), [
        Bot(Race.Terran, self.bot_data[0](self.bot_data[1], self.com_args)),
        Computer(Race.Protoss, Difficulty.Medium)
      ], realtime=False)

    if self.game is not None:
      self.com.notify()
      self.game.join()
      self.com.reset()

    self.game = Process(target=_game_process, args=(self,))
    self.game.start()

    return np.zeros(shape=self.base_plan["env_state_shape"], dtype=np.float64)


  def step(self, action: tuple[int]) -> np.array:
    self.com.send(action)
    self.state, self.reward, self.done = self.com.get()
    if self.graphics is not None:
      self.graphics.update(self.state)
    return self.state, self.reward[0], self.done[0], ""

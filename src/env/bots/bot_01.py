from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from .state_tools import get_uid

from ..com import ComClient
from .state_pipeline import pipeline as gen_state

import copy


class Bot(BotAI):
  def __init__(self, ai_args, com_args):
    self.view_size, self.map_size = ai_args
    self.com = ComClient(*com_args)

    self.cam = [[0, 0], [self.view_size[0], self.view_size[1]], [self.view_size[0] - 0, self.view_size[1] - 0]]
    self.selected_ = []

    self.state_, self.reward_, self.done_ = None, 1, False
    self.prev_self = None


  async def on_step(self, iteration):
    action = self.com.get()[0]

    await self.do_action(action)
    state, reward, done = (self.generate_state(),
                           self.generate_reward(),
                           self.generate_done())

    self.com.send(state, [reward], [done])
    self.prev_self = copy.copy(self)


  async def on_end(self, result):
    self.done_ = True
    _ = self.com.get()[0]
    state, reward, done = (self.generate_state(),
                           self.generate_reward(),
                           self.generate_done())

    self.com.send(state, [reward], [done])
    del self.com


  def generate_state(self):
    return gen_state(self, self.map_size, self.cam)


  def generate_reward(self):
    prev_reward = 0
    if self.prev_self is not None:
      for unit in self.prev_self.all_own_units:
        if unit.type_id == UnitTypeId.MARINE:
          prev_reward += 100
    
    reward = 0
    for unit in self.all_own_units:
      if unit.type_id == UnitTypeId.MARINE:
        reward += 100
    
    return reward - prev_reward - 1


  def generate_done(self):
    return self.done_


  async def do_action(self, action):
    await self.distribute_workers()
    match action:
      case 0:
        pass

      case 1:
        self.cam[0][0] = min(self.cam[0][0] + 1, self.map_size[0] - self.cam[2][0] - 1)
        self.cam[1][0] = self.cam[0][0] + self.cam[2][0]

      case 2:
        self.cam[0][0] = max(self.cam[0][0] - 1, 0)
        self.cam[1][0] = self.cam[0][0] + self.cam[2][0]

      case 3:
        self.cam[0][1] = min(self.cam[0][1] + 1, self.map_size[1] - self.cam[2][1] - 1)
        self.cam[1][1] = self.cam[0][1] + self.cam[2][1]

      case 4:
        self.cam[0][1] = max(self.cam[0][1] - 1, 0)
        self.cam[1][1] = self.cam[0][1] + self.cam[2][1]

      case 5:
        pointer_position = (self.cam[0][0] + self.cam[2][0] // 2,
                            self.cam[0][1] + self.cam[2][1] // 2)
        self.selected_ = [self.all_units.closest_to(Point2(pointer_position))]

      case 7:
        if len(self.selected_) > 0:
          selected_unit = self.selected_[0]
          pointer_position = (self.cam[0][0] + self.cam[2][0] // 2,
                              self.cam[0][1] + self.cam[2][1] // 2)
          if get_uid(selected_unit) == UnitTypeId.SCV:
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
              selected_unit.build(UnitTypeId.SUPPLYDEPOT, position=Point2(pointer_position))

      case 8:
        if len(self.selected_) > 0:
          selected_unit = self.selected_[0]
          pointer_position = (self.cam[0][0] + self.cam[2][0] // 2,
                              self.cam[0][1] + self.cam[2][1] // 2)
          if get_uid(selected_unit) == UnitTypeId.SCV:
            if self.can_afford(UnitTypeId.BARRACKS):
              selected_unit.build(UnitTypeId.BARRACKS, position=Point2(pointer_position))

      case 9:
        if len(self.selected_) > 0:
          selected_unit = self.selected_[0]
          pointer_position = (self.cam[0][0] + self.cam[2][0] // 2,
                              self.cam[0][1] + self.cam[2][1] // 2)
          if selected_unit.type_id == UnitTypeId.BARRACKS:
            if self.can_afford(UnitTypeId.MARINE):
              selected_unit.train(UnitTypeId.MARINE)

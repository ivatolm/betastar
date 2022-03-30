from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from .state_tools import get_uid, to_screen_pos, to_int_pos, to_abs_pos

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
    action = self.com.get()
    action = tuple(map(int, action))

    await self.do_action(action)
    state, reward, done = (self.generate_state(),
                           self.generate_reward(),
                           self.generate_done())

    self.com.send(state, [reward], [done])
    self.prev_self = copy.copy(self)


  async def on_end(self, result):
    self.done_ = True
    _ = self.com.get()
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
    cmd, args = action[0], action[1:]

    await self.distribute_workers()
    match cmd:
      case 0: # no_op
        pass

      # case 1: # move_cam
      #   x, y = args[:2]

      #   self.cam[0] = [self.cam[0][0] + x - self.cam[2][0] // 2, self.cam[0][1] + y - self.cam[2][1] // 2]
      #   self.cam[0] = [max(min(self.cam[0][0], self.map_size[0] - self.cam[2][0] - 1), 0),
      #                  max(min(self.cam[0][1], self.map_size[1] - self.cam[2][1] - 1), 0)]

      #   self.cam[1] = [self.cam[0][0] + self.cam[2][0], self.cam[0][1] + self.cam[2][1]]

      case 1: # select_rect
        x1, y1, x2, y2 = args
        if not (x1 < x2 and y1 < y2):
          return

        self.selected_ = []
        for unit in self.all_units:
          pos = to_int_pos(unit.position)
          screen_pos = to_screen_pos(self.cam, pos)
          if (x1 <= screen_pos[0] < x2) and (y1 <= screen_pos[1] < y2):
            self.selected_.append(unit)

      case 2: # build_supply
        x, y = args[:2]

        for unit in self.selected_:
          if get_uid(unit) == UnitTypeId.SCV:
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
              unit.build(UnitTypeId.SUPPLYDEPOT, position=Point2(to_abs_pos(self.cam, (x, y))))

      case 3: # build_barrack
        x, y = args[:2]

        for unit in self.selected_:
          if get_uid(unit) == UnitTypeId.SCV:
            if self.can_afford(UnitTypeId.BARRACKS):
              unit.build(UnitTypeId.BARRACKS, position=Point2(to_abs_pos(self.cam, (x, y))))
      
      case 4: # train_marine
        for unit in self.selected_:
          if get_uid(unit) == UnitTypeId.BARRACKS:
            if self.can_afford(UnitTypeId.MARINE):
              unit.train(UnitTypeId.MARINE)


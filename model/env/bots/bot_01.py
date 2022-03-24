from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from ..com import ComClient

import numpy as np
import copy


class Bot(BotAI):
  def __init__(self, ai_args, com_args):
    self.view_size, self.map_size = ai_args
    self.com = ComClient(*com_args)

    self.view_position = [self.map_size[0] // 2, self.map_size[1] // 2]
    self.pointer_position = [self.view_position[0] + self.view_size[0] // 2, self.view_position[1] + self.view_size[1] // 2]
    self.selected_unit = None

    self.state_, self.reward_, self.done_ = None, 1, False
    self.prev_state_, self.reward_prev_, self.prev_done_ = None, 1, False
    self.prev_self = None


  async def on_step(self, iteration):
    action = self.com.get()[0]

    await self.do_action(action)
    state, reward, done = (self.generate_state(),
                           self.generate_reward(),
                           self.generate_done())

    self.com.send(state, [reward], [done])
    self.prev_state_, self.reward_prev_, self.prev_done_ = state, reward, done
    self.prev_self = copy.copy(self)


  async def on_end(self, result):
    self.done_ = True
    _ = self.com.get()[0]
    state, reward, done = (self.generate_state(),
                           self.generate_reward(),
                           self.generate_done())

    self.com.send(state, [reward], [done])
    self.com.close()


  def generate_state(self):
    pos_v = np.zeros(self.view_size)
    selected_v = np.zeros(self.view_size)
    resource_v = np.zeros(self.view_size)
    unit_v = np.zeros(self.view_size)
    density_v = np.zeros(self.view_size)
    active_v = np.zeros(self.view_size)
    

    is_seeable = lambda x, y: ((0 <= x < self.view_size[0])
                  and
                   (0 <= y < self.view_size[1]))
    to_abs_coords = lambda x, y: ((x / self.map_size[0]) * self.view_size[0],
                    (y / self.map_size[1]) * self.view_size[1])

    # POS_V
    abs_view_pos = to_abs_coords(*self.view_position)
    pos_v[int(abs_view_pos[1])][int(abs_view_pos[0])] = 100
    if self.selected_unit is not None:
      unit_id = int(self.selected_unit.type_id)
      abs_pos = to_abs_coords(*self.selected_unit.position)
      pos_v[int(abs_pos[1])][int(abs_pos[0])] = unit_id

    # SELECTED_V
    if self.selected_unit is not None:
      unit_id = int(self.selected_unit.type_id)
      pos = (self.selected_unit.position[0] - self.view_position[0],
           self.selected_unit.position[1] - self.view_position[1])
      size = self.selected_unit.footprint_radius
      if size is not None:
        for y in range(int(2 * size)):
          for x in range(int(2 * size)):
            spos = (pos[0] + x - size, pos[1] + y - size)
            if is_seeable(*spos):
              selected_v[int(spos[1])][int(spos[0])] = unit_id
        if is_seeable(*pos) and size == 0.:
          selected_v[int(pos[1])][int(pos[0])] = unit_id

    # UNIT_V, DENSITY_V, ACTIVE_V
    for unit in self.all_own_units:
      unit_id = int(unit.type_id)
      pos = (unit.position[0] - self.view_position[0],
           unit.position[1] - self.view_position[1])
      size = unit.footprint_radius
      if size is not None:
        for y in range(int(2 * size)):
          for x in range(int(2 * size)):
            spos = (pos[0] + x - size, pos[1] + y - size)
            if is_seeable(*spos):
              unit_v[int(spos[1])][int(spos[0])] = unit_id
              density_v[int(spos[1])][int(spos[0])] += unit_id
              if unit.is_ready:
                active_v[int(spos[1])][int(spos[0])] = 100
              if unit.is_idle:
                active_v[int(spos[1])][int(spos[0])] = 200
              if unit.is_active:
                active_v[int(spos[1])][int(spos[0])] = 300
        if is_seeable(*pos) and size == 0.:
          unit_v[int(pos[1])][int(pos[0])] = unit_id
          density_v[int(pos[1])][int(pos[0])] += unit_id
          if unit.is_ready:
            active_v[int(pos[1])][int(pos[0])] = 100
          if unit.is_idle:
            active_v[int(pos[1])][int(pos[0])] = 200
          if unit.is_active:
            active_v[int(pos[1])][int(pos[0])] = 300

    # RESOURCE_V
    for unit in self.all_units.mineral_field:
      unit_id = int(unit.type_id)
      pos = (unit.position[0] - self.view_position[0],
           unit.position[1] - self.view_position[1])
      if is_seeable(*pos):
        resource_v[int(pos[1])][int(pos[0])] = unit_id

    pos_v = np.flipud(pos_v)
    selected_v = np.flipud(selected_v)
    resource_v = np.flipud(resource_v)
    unit_v = np.flipud(unit_v)
    density_v = np.flipud(density_v)
    active_v = np.flipud(active_v)

    self.state_ = (pos_v, selected_v, resource_v, unit_v, density_v, active_v)

    return self.state_


  def generate_reward(self):
    prev_reward = 0
    if self.prev_self is not None:
      for unit in self.prev_self.all_own_units:
        if unit.type_id == UnitTypeId.MARINE:
          prev_reward += 1
    
    reward = 0
    for unit in self.all_own_units:
      if unit.type_id == UnitTypeId.MARINE:
        reward += 1
    
    return reward - prev_reward - 1


  def generate_done(self):
    return self.done_


  async def do_action(self, action):
    await self.distribute_workers()
    match action:
      case 0:
        pass
      case 1:
        self.view_position[0] = min(self.view_position[0] + 1, self.map_size[0] - 1)
        self.pointer_position = [self.view_position[0] + self.view_size[0] // 2,
                     self.view_position[1] + self.view_size[1] // 2]
      case 2:    
        self.view_position[1] = min(self.view_position[1] + 1, self.map_size[1] - 1)
        self.pointer_position = [self.view_position[0] + self.view_size[0] // 2,
                     self.view_position[1] + self.view_size[1] // 2]
      case 3:
        self.view_position[0] = max(self.view_position[0] - 1, 10)
        self.pointer_position = [self.view_position[0] + self.view_size[0] // 2,
                     self.view_position[1] + self.view_size[1] // 2]
      case 4:
        self.view_position[1] = max(self.view_position[1] - 1, 10)
        self.pointer_position = [self.view_position[0] + self.view_size[0] // 2,
                     self.view_position[1] + self.view_size[1] // 2]
      case 5:
        self.selected_unit = self.all_units.closest_to(Point2(self.pointer_position))
      case 6:
        if self.selected_unit is not None:
          self.selected_unit.move(Point2(self.pointer_position))
      case 7:
        if self.selected_unit is not None:
          if self.selected_unit.type_id == UnitTypeId.SCV:
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
              self.selected_unit.build(UnitTypeId.SUPPLYDEPOT, position=Point2(self.pointer_position))
      case 8:
        if self.selected_unit is not None:
          if self.selected_unit.type_id == UnitTypeId.SCV:
            if self.can_afford(UnitTypeId.BARRACKS):
              self.selected_unit.build(UnitTypeId.BARRACKS, position=Point2(self.pointer_position))
      case 9:
        if self.selected_unit is not None:
          if self.selected_unit.type_id == UnitTypeId.BARRACKS:
            if self.can_afford(UnitTypeId.MARINE):
              self.selected_unit.train(UnitTypeId.MARINE)

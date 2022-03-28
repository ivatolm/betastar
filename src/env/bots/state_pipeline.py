from .state_tools import *

from sc2.ids.unit_typeid import UnitTypeId
import numpy as np


MAX_UID = int(max(UnitTypeId))
MAX_STATUS = 2


def pipeline(bot_data, map_size, cam):
  return np.array([
    gen_unit_type_view(bot_data, cam),
    gen_unit_status_view(bot_data, cam),
    gen_unit_selected_view(bot_data, cam),
    gen_minimap_camera_view(map_size, cam)
  ])


def gen_unit_type_view(bot_data, cam):
  view = np.zeros(cam[2])

  for unit in bot_data.all_units:
    pos = to_int_pos(unit.position)
    uid = get_uid(unit)
    size = get_size(unit)

    for (x, y) in get_seeable_parts(cam, pos, size):
      view[y][x] = uid / MAX_UID

  return view


def gen_unit_status_view(bot_data, cam):
  view = np.zeros(cam[2])

  for unit in bot_data.all_units:
    pos = to_int_pos(unit.position)
    size = get_size(unit)

    for (x, y) in get_seeable_parts(cam, pos, size):
      status = 0.00
      if not unit.is_ready:
        status = unit.build_progress
      if unit.is_idle:
        status = 0.01
      if unit.is_active:
        status = 0.02

      view[y][x] = status / MAX_STATUS

  return view


def gen_unit_selected_view(bot_data, cam):
  view = np.zeros(cam[2])

  for unit in bot_data.all_units:
    if unit not in bot_data.selected_:
      continue

    pos = to_int_pos(unit.position)
    size = get_size(unit)

    for (x, y) in get_seeable_parts(cam, pos, size):
      view[y][x] = 0.01

  return view


def gen_minimap_camera_view(map_size, cam):
  view = np.zeros(cam[2])

  scale_factor = ((cam[2][0] - 1) / map_size[0], (cam[2][1] - 1) / map_size[1])
  scaled_cam = (to_int_pos((cam[0][0] * scale_factor[0],
                            cam[0][1] * scale_factor[1])),
                to_int_pos((cam[1][0] * scale_factor[0],
                            cam[1][1] * scale_factor[1])))

  for y in range(scaled_cam[0][1], scaled_cam[1][1]):
    for x in range(scaled_cam[0][0], scaled_cam[1][0]):
      view[y][x] = 0.01

  return view

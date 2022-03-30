def to_int_pos(position):
  return (int(position[0]), int(position[1]))


def to_screen_pos(cam, position):
  return (position[0] - cam[0][0],
          position[1] - cam[0][1])


def to_abs_pos(cam, position):
  return (position[0] + cam[0][0],
          position[1] + cam[0][1])


def get_seeable_parts(cam, position, size):
  seeable = []
  for y in range(size):
    for x in range(size):
      pos = (position[0] - size // 2 + x, position[1] - size // 2 + y)
      if ((cam[0][0] <= pos[0] < cam[1][0])
           and
          (cam[0][1] <= pos[1] < cam[1][1])):
        seeable.append(to_screen_pos(cam, pos))
  return seeable


def get_uid(unit):
  return int(unit.type_id)


def get_size(unit):
  size = int(2 * unit.radius)
  return size if size != 0 else 1

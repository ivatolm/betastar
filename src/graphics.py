from collections import deque
import pygame
import colorsys

from configs.graphics_cfg import *
from configs.pipeline_cfg import *


class Graphics:
  def __init__(self):
    pygame.init()
    self.screen = pygame.display.set_mode(DISPLAY_SIZE)
    self.font = pygame.font.SysFont('Roboto Mono Bold', FONT_SIZE)


  def render_view(self, position, view):
    for y, line in enumerate(view):
        for x, item in enumerate(line):
          normalized_item = min(item, 359) / 360.
          color = self._hsv2rgb(normalized_item, 1, min(1, normalized_item + 0.1))
          pygame.draw.rect(self.screen, color,
                           (position[0] + x * CELL_SIZE, position[1] + y * CELL_SIZE,
                           CELL_SIZE, CELL_SIZE))


  def update(self, state, action):
    self.screen.fill(BACKGROUND_COLOR)

    for i, view in enumerate(state):
      position = (
        (i % VIEWS_IN_ROW) * (OFFSET_SIZE + CELL_SIZE * ENV_VIEW_SIZE[0]),
        (i // VIEWS_IN_ROW) * (OFFSET_SIZE + CELL_SIZE * ENV_VIEW_SIZE[1])
      )
      self.render_view(position, view)

    pygame.display.update()


  def _hsv2rgb(self, h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

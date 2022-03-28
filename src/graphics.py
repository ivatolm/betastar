import pygame
import colorsys
import random

from configs.graphics_cfg import *
from configs.pipeline_cfg import *


class Graphics:
  def __init__(self):
    pygame.init()
    self.screen = pygame.display.set_mode(DISPLAY_SIZE)
    self.font = pygame.font.SysFont('Iosevka', FONT_SIZE)
    self.colors = {}
    self.prev_q = None


  def render_view(self, position, view):
    for y, line in enumerate(view):
        for x, item in enumerate(line):
          if item in self.colors:
            color = self.colors[item]
          else:
            color = random.random()
            self.colors[item] = color
          
          rgb_color = self._hsv2rgb(color, 1, 1)
          if item == 0:
            rgb_color = (20, 20, 20)
          pygame.draw.rect(self.screen, rgb_color,
                           (position[0] + x * CELL_SIZE, position[1] + y * CELL_SIZE,
                           CELL_SIZE, CELL_SIZE))


  def render_status_bar(self, q):
    text = self.font.render(f"Estimated Q: {q}", True, FONT_COLOR)
    self.screen.blit(text, (0, VIEWS_SIZE[1]))


  def update(self, state, action, q):
    self.screen.fill(BACKGROUND_COLOR)

    for i, view in enumerate(state):
      position = (
        (i % VIEWS_IN_ROW) * (OFFSET_SIZE + CELL_SIZE * ENV_VIEW_SIZE[0]),
        (i // VIEWS_IN_ROW) * (OFFSET_SIZE + CELL_SIZE * ENV_VIEW_SIZE[1])
      )
      self.render_view(position, view)

    if q is not None:
      self.prev_q = q

    if self.prev_q is not None:
      self.render_status_bar(self.prev_q)


    pygame.display.update()


  def _hsv2rgb(self, h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

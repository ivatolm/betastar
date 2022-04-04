import pygame
import colorsys
import random
import math

from configs.graphics_cfg import *

class Graphics:
  def __init__(self):
    pygame.init()
    self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
    self.prev_q = None
    self.colors = {}

    self.layout = None
    self.views_size = None
    self.statusbar_size = None
    self.display_size = None

    self.screen = None


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
    self.screen.blit(text, (0, self.views_size[1]))


  def update(self, state):
    if self.screen is None:
      channels_num = state.shape[0]
      view_size = state[0].shape
      self.layout = (math.ceil(channels_num / VIEWS_IN_ROW),
                     min(VIEWS_IN_ROW, channels_num))
      self.views_size = (self.layout[1] * (view_size[0] * CELL_SIZE) + (self.layout[1] - 1) * OFFSET_SIZE,
                         self.layout[0] * (view_size[1] * CELL_SIZE) + (self.layout[0] - 1) * OFFSET_SIZE)
      self.display_size = (self.views_size[0],
                           self.views_size[1])
      self.screen = pygame.display.set_mode(self.display_size)

    self.screen.fill(BACKGROUND_COLOR)

    for i, view in enumerate(state):
      shape = view.shape
      position = (
        (i % VIEWS_IN_ROW) * (OFFSET_SIZE + CELL_SIZE * shape[0]),
        (i // VIEWS_IN_ROW) * (OFFSET_SIZE + CELL_SIZE * shape[1])
      )
      self.render_view(position, view)

    pygame.display.update()


  def _hsv2rgb(self, h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

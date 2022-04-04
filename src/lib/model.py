import torch
import numpy as np
import torch.nn as nn

from ..types import Model_t
from configs.torch_cfg import DEVICE


class DQNModel(Model_t):
  def __init__(self, input_shape: tuple[int], output_shape: tuple[int]) -> None:
    super(Model_t, self).__init__()

    self.input_shape  = input_shape
    self.output_shape = output_shape

    self.f = nn.Sequential(
      nn.Linear(input_shape[0], 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, output_shape[0])
    )


  def forward(self, x: torch.tensor) -> torch.tensor:
    x = x.to(DEVICE)

    x = self.f(x)
    x = x.double()

    return x


class NdimDDQNModel(Model_t):
  def __init__(self, input_shape: tuple[int], output_shapes: tuple[tuple[int]]) -> None:
    super(Model_t, self).__init__()

    self.input_shape   = input_shape
    self.output_shapes = output_shapes

    self.conv = nn.Sequential(
      nn.Conv2d(input_shape[0], 256, 8),
      nn.ReLU(),
      nn.Conv2d(256, 256, 4),
      nn.ReLU(),
      nn.Conv2d(256, 256, 2),
      nn.ReLU()
    ).to(DEVICE)

    self.flatten = nn.Flatten().to(DEVICE)

    conv_out_size = self._get_conv_out_size(input_shape)

    self.linear = nn.Sequential(
      nn.Linear(conv_out_size, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
    ).to(DEVICE)

    self.fv = nn.Sequential(
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 1)
    ).to(DEVICE)

    self.fa = []
    for shape in output_shapes:
      self.fa.append(nn.Sequential(nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, shape[0])).to(DEVICE))


  def forward(self, x: torch.tensor) -> tuple[torch.tensor]:
    x = x.float()
    x = x.to(DEVICE)

    x = x.view(-1, *self.input_shape)
    x = self.conv(x)
    x = self.flatten(x)
    x = self.linear(x)

    value = self.fv(x)
    qs_a  = []
    for fa in self.fa:
      advantages = fa(x)
      qs         = value + advantages - advantages.mean()
      qs         = qs.double()
      qs_a.append(qs)

    return tuple(qs_a)


  def _get_conv_out_size(self, input_shape: tuple[int]) -> int:
    input_t  = torch.zeros(1, *input_shape).to(DEVICE)
    output_t = self.conv(input_t)
    return int(np.prod(output_t.size()))

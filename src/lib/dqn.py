import torch
import torch.nn as nn
import numpy as np

from configs.torch_cfg import *


class DQN(nn.Module):
	def __init__(self, input_shape, output_shapes):
		super(DQN, self).__init__()

		self.input_shape = input_shape
		self.output_shapes = output_shapes

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_shape[0], 32, 8, 4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4),
			nn.ReLU(),
			nn.Conv2d(64, 64, 2),
			nn.ReLU(),
		).to(DEVICE)

		conv_out_size = self._get_conv_out(input_shape)

		self.flatten = nn.Flatten().to(DEVICE)

		self.fv = nn.Sequential(
			nn.Linear(conv_out_size, 512),
			nn.ReLU(),
			nn.Linear(512, 1)
		).to(DEVICE)

		self.fa = []
		for shape in output_shapes:
			self.fa.append(nn.Sequential(nn.Linear(conv_out_size, 512),
																	 nn.ReLU(),
																	 nn.Linear(512, shape[0])).to(DEVICE))


	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape).to(DEVICE))
		return int(np.prod(o.size()))


	def forward(self, x):
		x = x.float()
		x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])

		x = self.conv(x)
		x = self.flatten(x)

		value = self.fv(x)
		qs = []
		for fa in self.fa:
			advantages = fa(x)
			qs.append(value + advantages - advantages.mean())

		return qs

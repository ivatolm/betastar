import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
	def __init__(self, input_shape, output_shape):
		super(DQN, self).__init__()

		self.input_shape = input_shape
		self.output_shape = output_shape

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_shape[0], 32, 8, 4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 2, 1, 1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3),
			nn.ReLU(),
		)

		conv_out_size = self._get_conv_out(input_shape)
		self.f_actions = nn.Sequential(
				nn.Flatten(),
				nn.Linear(conv_out_size, 512),
				nn.ReLU(),
				nn.Linear(512, output_shape[0])
		)


	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		return int(np.prod(o.size()))


	def forward(self, x):
		x = x.float()
		x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])

		actions = self.f_actions(self.conv(x))

		return actions

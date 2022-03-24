import torch.nn as nn


class DQN(nn.Module):
	def __init__(self, input_shape, output_shape):
		super(DQN, self).__init__()

		self.input_shape = input_shape
		self.output_shape = output_shape

		self.f_conv = nn.Sequential(
			nn.Conv2d(self.input_shape[0], 32, 4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4),
			nn.ReLU(),
			nn.Conv2d(64, 128, 4),
			nn.ReLU(),

			nn.Flatten(),
			nn.Linear(128 * (input_shape[1] - (4 + 4 + 4) + 3) * (input_shape[2] - (4 + 4 + 4) + 3), 512),
			nn.ReLU(),
		)

		self.f_hidden = nn.Sequential(
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 32),
			nn.ReLU(),
		)

		self.f_actions = nn.Sequential(
			nn.Linear(32, output_shape[0])
		)

	def forward(self, x):
		x = x.float()
		x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])

		conv = self.f_conv(x)
		hidden = self.f_hidden(conv)
		
		actions = self.f_actions(hidden)

		return actions
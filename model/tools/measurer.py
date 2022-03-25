from torch.utils.tensorboard import SummaryWriter

from configs.data_cfg import *


class Measurer:
  def __init__(self, label):
    self.label = label
    self.writer = SummaryWriter(MEASUREMENT_DIR + '/' + START_TIMESTAMP)


  def add_value(self, metric_name, value, index):
    self.writer.add_scalar(self.label + '/' + metric_name, value, index)


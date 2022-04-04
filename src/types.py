import torch
import torch.nn as nn
import numpy as np


class Model_t(nn.Module):
  def __init__(self, input_shape, output_shape) -> None:
    raise NotImplementedError


  def forward(self, x) -> None:
    raise NotImplementedError



class Policy_t:
  def __init__(self) -> None:
    raise NotImplementedError
  
  
  def get_action(self) -> int:
    raise NotImplementedError
  

  def update(self) -> None:
    raise NotImplementedError


class Memory_t:
  def __init__(self) -> None:
    raise NotImplementedError


  def push(self) -> None:
    raise NotImplementedError


  def sample(self) -> None:
    raise NotImplementedError


  def size(self) -> int:
    raise NotImplementedError


class Env_t:
  def __init__(self) -> None:
    raise NotImplementedError


  def reset(self) -> None:
    raise NotImplementedError


  def step(self) -> None:
    raise NotImplementedError


class Loss_t:
  def __init__(self) -> None:
    raise NotImplementedError


  def get_loss(self, batch: np.array, model: Model_t, target_model: Model_t, gamma: float, loss_func_name: str) -> torch.tensor:
    raise NotImplementedError

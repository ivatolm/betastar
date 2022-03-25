def train_cycle(net, target_net, memory, optimizer, loss_func, gamma, batch_size):
  optimizer.zero_grad()
  batch = memory.sample(batch_size)
  loss_t = loss_func(batch, net, target_net, gamma)
  loss_t.backward()
  optimizer.step()

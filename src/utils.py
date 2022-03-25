def train_cycle(net, target_net, memory, optimizer, loss_func, batch_size, steps, gamma):
  optimizer.zero_grad()
  batch = memory.sample(batch_size, steps)
  loss_t = loss_func(batch, net, target_net, gamma)
  loss_t.backward()
  optimizer.step()

def train_cycle(net, target_net, memory, optimizer, loss_func, batch_size, steps, gamma):
  optimizer.zero_grad()
  batch = memory.sample(batch_size, steps)
  loss_t = loss_func(batch, net, target_net, gamma)
  loss_t.backward()
  for param in net.parameters():
    param.grad.data.clamp_(-1, 1)
  optimizer.step()
  return loss_t.item()


def gen_plan_str(plan_name, plan):
  res = f"{plan_name} config:\n"
  for key, item in plan.items():
    res += f"  {key}: {item}\n"
  return res

import torch


x1 = torch.Tensor([2.0]).double()
x1.requires_grad = True

x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True

w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True

w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True

b = torch.Tensor([6.8]).double()
b.requires_grad = True

n = w1 * x1 + w2 * x2 + b
out = torch.tanh(n)

print(out.data.item())

out.backward()
print(f'x1 = {x1.grad.item()}')
print(f'x2 = {x2.grad.item()}')
print(f'w1 = {w1.grad.item()}')
print(f'w2 = {w2.grad.item()}')

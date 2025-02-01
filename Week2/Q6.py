import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

a = 2 * x
b = torch.sin(y)
c = a / b
d = c * z
e = torch.log(d + 1)
f = torch.tanh(e)
f.backward()

print(f"Gradient of f with respect to y: {y.grad.item()}")
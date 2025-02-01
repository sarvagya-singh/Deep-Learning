import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
x = 2 * a + 3 * b
y = 5 * a*a + 3 * b*b*b
z = 2 * x + 3 * y

z.backward()

print(f" dz/da : {a.grad.item()}")
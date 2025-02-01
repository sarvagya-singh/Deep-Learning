import torch

def func(x):
    return 8*x**4 + 3*x**3 + 7*x**2 + 6*x + 3

def analytical_gradient(x):
    return 32*x**3 + 9*x**2 + 14*x + 6

x = torch.tensor(float(input("Enter a value for x: ")), requires_grad=True)

y = func(x)

analytical_grad = analytical_gradient(x)

y.backward()

print(f"Analytical gradient at x = {x.item()} is: {analytical_grad.item()}")
print(f"PyTorch computed gradient at x = {x.item()} is: {x.grad.item()}")
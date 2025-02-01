import torch

x = torch.tensor(2.0, requires_grad=True)
def analytical(x) :
    internal = -(x * x) - (2 * x) - torch.sin(x)
    f = torch.exp(internal)
    grad = -f * (2 * x + 2 + torch.cos(x))
    return grad
print("Analytical Method : ", analytical(x))
internal = -(x*x) - (2*x) - torch.sin(x)
f = torch.exp(internal)

f.backward()
print("AutoGrad : ", x.grad)
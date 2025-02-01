import torch

w = torch.tensor(float(input("Enter Float Value for w : "))).requires_grad_(True)
x = torch.tensor(float(input("Enter Float Value for x : "))).requires_grad_(True)
b = torch.tensor(float(input("Enter Float Value for b : "))).requires_grad_(True)

def comp_a(w,x,b) :
    return w*x + b
def comp_grad(v) :
    if v > 0:
        # If v > 0, ReLU derivative is 1, and we get the gradient as x
        grad_w = x
    else:
        # If v <= 0, ReLU derivative is 0, and gradient is 0
        grad_w = 0
    return grad_w
a_analytical = comp_a(w,x,b)
a_grad = comp_grad(a_analytical)
print(a_grad)
u = w*x
v = u + b
a = torch.relu(v)

a.backward()
print(w.grad)

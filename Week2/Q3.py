import torch

w = torch.tensor(float(input("Enter Float Value for w : "))).requires_grad_(True)
x = torch.tensor(float(input("Enter Float Value for x : "))).requires_grad_(True)
b = torch.tensor(float(input("Enter Float Value for b : "))).requires_grad_(True)

def comp_a(w,x,b) :
    return torch.sigmoid(w*x + b)
def comp_grad(v) :
    return v*(1-v)
a_analytical = comp_a(w,x,b)
a_grad = comp_grad(a_analytical) * x
print(a_grad)
u = w*x
v = u + b
a = torch.sigmoid(v)

a.backward()
print(w.grad)

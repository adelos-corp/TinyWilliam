import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print("x =", x)
print("y =", y)
print("gradient =", x.grad)

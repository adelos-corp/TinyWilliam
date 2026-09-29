import torch

z = torch.tensor([
    -5.0,
    -2.0,
    0.0,
    2.0,
    5.0
])

p = torch.sigmoid(z)

derivative = p * (1 - p)

print("z:")
print(z)

print("\nsigmoid(z):")
print(p)

print("\nsigmoid derivative:")
print(derivative)
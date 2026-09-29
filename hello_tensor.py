import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Using:", device)

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
], device=device)

y = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]
], device=device)

z = x @ y

print("\nX:")
print(x)

print("\nY:")
print(y)

print("\nX @ Y:")
print(z)

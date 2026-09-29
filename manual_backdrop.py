import torch

# -------------------------
# Data
# -------------------------

X = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

# -------------------------
# Parameters
# -------------------------

W = torch.tensor([
    [0.5, -0.5],
    [1.0,  1.0]
])

b = torch.tensor([0.0, 0.0])

# -------------------------
# Forward pass
# -------------------------

Z = X @ W + b

A = torch.relu(Z)

print("X:")
print(X)

print("\nW:")
print(W)

print("\nZ:")
print(Z)

print("\nA:")
print(A)

# -------------------------
# Target
# -------------------------

Y = torch.tensor([
    [3.0, 2.0],
    [6.0, 4.0]
])

# Mean squared error
loss = ((A - Y) ** 2).mean()

print("\nY:")
print(Y)

print("\nLoss:")
print(loss)

# -------------------------
# Manual backward pass
# -------------------------

N = Y.numel()

# Gradient of loss with respect to A
dA = 2 * (A - Y) / N

# ReLU gradient
dZ = dA * (Z > 0)

# Gradient with respect to W
dW = X.T @ dZ

# Gradient with respect to b
db = dZ.sum(dim=0)

print("\ndA:")
print(dA)

print("\ndZ:")
print(dZ)

print("\ndW:")
print(dW)

print("\ndb:")
print(db)

# -------------------------
# PyTorch verification
# -------------------------

W_auto = W.clone().detach().requires_grad_(True)
b_auto = b.clone().detach().requires_grad_(True)

Z_auto = X @ W_auto + b_auto
A_auto = torch.relu(Z_auto)

loss_auto = ((A_auto - Y) ** 2).mean()

loss_auto.backward()

print("\nPyTorch dW:")
print(W_auto.grad)

print("\nPyTorch db:")
print(b_auto.grad)

print("\nManual dW == PyTorch dW:")
print(torch.allclose(dW, W_auto.grad))

print("\nManual db == PyTorch db:")
print(torch.allclose(db, b_auto.grad))
dX = dZ @ W.T

print("\ndX:")
print(dX)
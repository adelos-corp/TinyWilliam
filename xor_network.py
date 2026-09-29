import torch

# -------------------------
# XOR data
# -------------------------

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


# -------------------------
# Parameters
# -------------------------

W1 = torch.randn(2, 4) * 0.5
b1 = torch.zeros(4)

W2 = torch.randn(4, 1) * 0.5
b2 = torch.zeros(1)

learning_rate = 0.05


# -------------------------
# Training
# -------------------------

for step in range(10000):

    # ===== FORWARD =====

    Z1 = X @ W1 + b1
    A1 = torch.relu(Z1)

    Y_pred = A1 @ W2 + b2

    # ===== LOSS =====

    loss = ((Y_pred - Y) ** 2).mean()

    # ===== BACKWARD =====

    N = Y.numel()

    dY_pred = 2 * (Y_pred - Y) / N

    dW2 = A1.T @ dY_pred
    db2 = dY_pred.sum(dim=0)

    dA1 = dY_pred @ W2.T

    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = dZ1.sum(dim=0)

    # ===== UPDATE =====

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if step % 1000 == 0:
        print(
            f"Step {step:5d} | "
            f"Loss: {loss.item():.8f}"
        )


# -------------------------
# Results
# -------------------------

print("\nPredictions:")
print(Y_pred)

print("\nExpected:")
print(Y)

print("\nFinal loss:")
print(loss.item())
import torch

# -------------------------
# Training data
# -------------------------

X = torch.tensor([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 2.0]
])

Y = torch.tensor([
    [6.0],
    [8.0],
    [9.0],
    [13.0]
])


# -------------------------
# Parameters
# -------------------------

W1 = torch.randn(2, 3) * 0.1
b1 = torch.zeros(3)

W2 = torch.randn(3, 1) * 0.1
b2 = torch.zeros(1)

learning_rate = 0.01


# -------------------------
# Training
# -------------------------

for step in range(5000):

    # ===== FORWARD =====

    Z1 = X @ W1 + b1

    A1 = torch.relu(Z1)

    Y_pred = A1 @ W2 + b2

    loss = ((Y_pred - Y) ** 2).mean()


    # ===== BACKWARD =====

    N = Y.numel()

    # Loss → output
    dY_pred = 2 * (Y_pred - Y) / N

    # Output layer
    dW2 = A1.T @ dY_pred
    db2 = dY_pred.sum(dim=0)

    # Gradient entering hidden layer
    dA1 = dY_pred @ W2.T

    # Through ReLU
    dZ1 = dA1 * (Z1 > 0)

    # Hidden layer
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(dim=0)


    # ===== UPDATE =====

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


    # ===== PRINT =====

    if step % 500 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.8f}"
        )


print("\nPredictions:")
print(Y_pred)

print("\nFinal loss:")
print(loss.item())

# -------------------------
# Test on unseen data
# -------------------------

X_test = torch.tensor([
    [4.0, 3.0],
    [5.0, 1.0],
    [2.0, 5.0]
])

Y_test = X_test @ W1
Y_test = torch.relu(Y_test)
Y_test = Y_test @ W2 + b2

print("\nUnseen inputs:")
print(X_test)

print("\nPredictions:")
print(Y_test)
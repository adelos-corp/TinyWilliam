import torch

# Input data
X = torch.tensor([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 2.0]
])

# Desired outputs
y_true = torch.tensor([
    [6.0],
    [8.0],
    [9.0],
    [13.0]
])

# Layer 1: 2 inputs -> 3 neurons
W1 = torch.randn(2, 3, requires_grad=True)
b1 = torch.zeros(3, requires_grad=True)

# Layer 2: 3 neurons -> 1 output
W2 = torch.randn(3, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

learning_rate = 0.01

for step in range(2000):

    # Layer 1
    z1 = X @ W1 + b1

    # Activation
    h = torch.relu(z1)

    # Layer 2
    y_pred = h @ W2 + b2

    # Loss
    loss = ((y_pred - y_true) ** 2).mean()

    # Backpropagation
    loss.backward()

    # Gradient descent
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad

        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

    # Reset gradients
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    if step % 200 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f}"
        )

print("\nFinal predictions:")
print(y_pred)

print("\nFinal loss:")
print(loss.item())
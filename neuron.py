import torch

# Training data
X = torch.tensor([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 2.0]
])

y_true = torch.tensor([
    6.0,
    8.0,
    9.0,
    13.0
])

# Trainable parameters
w1 = torch.tensor(0.0, requires_grad=True)
w2 = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01

for step in range(1000):

    # Neuron
    y_pred = w1 * X[:, 0] + w2 * X[:, 1] + b

    # Mean squared error
    loss = ((y_pred - y_true) ** 2).mean()

    # Calculate gradients
    loss.backward()

    # Update parameters
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        b -= learning_rate * b.grad

    # Reset gradients
    w1.grad.zero_()
    w2.grad.zero_()
    b.grad.zero_()

    if step % 100 == 0:
        print(
            f"Step {step:3d} | "
            f"Loss: {loss.item():.6f} | "
            f"w1: {w1.item():.4f} | "
            f"w2: {w2.item():.4f} | "
            f"b: {b.item():.4f}"
        )

print("\nFinal parameters:")
print("w1 =", w1.item())
print("w2 =", w2.item())
print("b  =", b.item())
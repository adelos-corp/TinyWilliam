import torch

# Our training data
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])

# The model's parameter
weight = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.04132

for step in range(100):

    # Model prediction
    y_pred = weight * x

    # How wrong are we?
    loss = ((y_pred - y_true) ** 2).mean()

    # Calculate gradient
    loss.backward()

    # Update the weight
    with torch.no_grad():
        weight -= learning_rate * weight.grad

    # Reset gradient
    weight.grad.zero_()

    print(
        f"Step {step:3d} | "
        f"Loss: {loss.item():10.6f} | "
        f"Weight: {weight.item():10.6f}"
    )

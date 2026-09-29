import torch

# -------------------------
# Token IDs
# -------------------------

tokens = torch.tensor([0, 1, 2, 3])

# 1 = animal
# 0 = object
targets = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
])

# -------------------------
# Embedding matrix
# -------------------------

embedding = torch.randn(4, 3) * 0.1
embedding.requires_grad_()

# Linear classifier
W = torch.randn(3, 1) * 0.1
W.requires_grad_()

b = torch.zeros(1, requires_grad=True)

learning_rate = 0.1

# -------------------------
# Training
# -------------------------

for step in range(1000):

    # Embedding lookup
    E = embedding[tokens]

    # Linear layer
    logits = E @ W + b

    # Probability
    predictions = torch.sigmoid(logits)

    # Binary cross-entropy
    loss = -(
        targets * torch.log(predictions)
        + (1 - targets) * torch.log(1 - predictions)
    ).mean()

    # Backpropagation
    loss.backward()

    # Update
    with torch.no_grad():
        embedding -= learning_rate * embedding.grad
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    # Reset gradients
    embedding.grad.zero_()
    W.grad.zero_()
    b.grad.zero_()

    if step % 100 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f}"
        )

# -------------------------
# Results
# -------------------------

print("\nFinal predictions:")
print(predictions)

print("\nFinal embeddings:")
print(embedding)
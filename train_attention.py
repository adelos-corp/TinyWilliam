import torch

torch.manual_seed(42)

# --------------------------------------------------
# Tiny sequence
# --------------------------------------------------

X = torch.tensor([0, 1, 2, 3])
Y = torch.tensor([1, 2, 3, 0])

vocab_size = 4
embedding_dim = 4

# --------------------------------------------------
# Learnable embeddings
# --------------------------------------------------

embedding = torch.randn(vocab_size, embedding_dim) * 0.1
embedding.requires_grad_()

# --------------------------------------------------
# Learnable attention projections
# --------------------------------------------------

W_Q = torch.randn(embedding_dim, embedding_dim) * 0.1
W_K = torch.randn(embedding_dim, embedding_dim) * 0.1
W_V = torch.randn(embedding_dim, embedding_dim) * 0.1

W_Q.requires_grad_()
W_K.requires_grad_()
W_V.requires_grad_()

# --------------------------------------------------
# Output projection
# --------------------------------------------------

W_O = torch.randn(embedding_dim, vocab_size) * 0.1
W_O.requires_grad_()

learning_rate = 0.05

# --------------------------------------------------
# Training
# --------------------------------------------------

for step in range(3000):

    # Embedding lookup
    E = embedding[X]

    # Q, K, V
    Q = E @ W_Q
    K = E @ W_K
    V = E @ W_V

    # Attention scores
    scores = Q @ K.T
    scores = scores / (embedding_dim ** 0.5)

    # Causal mask
    mask = torch.triu(
        torch.ones(len(X), len(X)),
        diagonal=1
    )

    scores = scores.masked_fill(
        mask == 1,
        float("-inf")
    )

    # Attention weights
    attention = torch.softmax(scores, dim=-1)

    # Attention output
    H = attention @ V

    # Predict next token
    logits = H @ W_O

    # Cross entropy
    loss = torch.nn.functional.cross_entropy(
        logits,
        Y
    )

    # Backpropagation
    loss.backward()

    # Update parameters
    with torch.no_grad():
        embedding -= learning_rate * embedding.grad
        W_Q -= learning_rate * W_Q.grad
        W_K -= learning_rate * W_K.grad
        W_V -= learning_rate * W_V.grad
        W_O -= learning_rate * W_O.grad

    # Clear gradients
    embedding.grad.zero_()
    W_Q.grad.zero_()
    W_K.grad.zero_()
    W_V.grad.zero_()
    W_O.grad.zero_()

    if step % 300 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f}"
        )

# --------------------------------------------------
# Inspect predictions
# --------------------------------------------------

predictions = torch.argmax(logits, dim=-1)

print("\nInput:")
print(X)

print("\nExpected next token:")
print(Y)

print("\nPredicted next token:")
print(predictions)

print("\nAttention:")
print(attention)
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ==================================================
# Configuration
# ==================================================

vocab_size = 4
d_model = 8
d_ff = 32
sequence_length = 8

learning_rate = 0.05
steps = 3000

# ==================================================
# Training data
# ==================================================

tokens = torch.tensor([
    0, 1, 2, 3,
    0, 1, 2, 3
])

inputs = tokens[:-1]
targets = tokens[1:]

# ==================================================
# Parameters
# ==================================================

embedding = torch.randn(vocab_size, d_model) * 0.1
embedding.requires_grad_()

position_embedding = torch.randn(
    sequence_length - 1,
    d_model
) * 0.1

position_embedding.requires_grad_()

W_Q = torch.randn(d_model, d_model) * 0.1
W_K = torch.randn(d_model, d_model) * 0.1
W_V = torch.randn(d_model, d_model) * 0.1

W_Q.requires_grad_()
W_K.requires_grad_()
W_V.requires_grad_()

# Feed-forward network

W1 = torch.randn(d_model, d_ff) * 0.1
b1 = torch.zeros(d_ff)

W2 = torch.randn(d_ff, d_model) * 0.1
b2 = torch.zeros(d_model)

W1.requires_grad_()
b1.requires_grad_()
W2.requires_grad_()
b2.requires_grad_()

# LayerNorm parameters

gamma1 = torch.ones(d_model, requires_grad=True)
beta1 = torch.zeros(d_model, requires_grad=True)

gamma2 = torch.ones(d_model, requires_grad=True)
beta2 = torch.zeros(d_model, requires_grad=True)

# Output projection

W_out = torch.randn(d_model, vocab_size) * 0.1
b_out = torch.zeros(vocab_size)

W_out.requires_grad_()
b_out.requires_grad_()

# ==================================================
# LayerNorm function
# ==================================================

def layer_norm(x, gamma, beta):

    mean = x.mean(dim=-1, keepdim=True)

    variance = (
        (x - mean) ** 2
    ).mean(dim=-1, keepdim=True)

    normalized = (
        x - mean
    ) / torch.sqrt(variance + 1e-5)

    return gamma * normalized + beta


# ==================================================
# Training
# ==================================================

for step in range(steps):

    # ----------------------------------------------
    # Embedding
    # ----------------------------------------------

    token_vectors = embedding[inputs]

    positions = torch.arange(
        len(inputs)
    )

    position_vectors = position_embedding[positions]

    X = token_vectors + position_vectors

    # ----------------------------------------------
    # Self-attention
    # ----------------------------------------------

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T

    scores = scores / (d_model ** 0.5)

    # Causal mask

    mask = torch.triu(
        torch.ones(
            len(inputs),
            len(inputs)
        ),
        diagonal=1
    )

    scores = scores.masked_fill(
        mask == 1,
        float("-inf")
    )

    attention = torch.softmax(
        scores,
        dim=-1
    )

    H = attention @ V

    # ----------------------------------------------
    # Residual + LayerNorm
    # ----------------------------------------------

    Z = layer_norm(
        X + H,
        gamma1,
        beta1
    )

    # ----------------------------------------------
    # Feed-forward network
    # ----------------------------------------------

    hidden = Z @ W1 + b1

    hidden = F.gelu(hidden)

    F_out = hidden @ W2 + b2

    # ----------------------------------------------
    # Residual + LayerNorm
    # ----------------------------------------------

    Y = layer_norm(
        Z + F_out,
        gamma2,
        beta2
    )

    # ----------------------------------------------
    # Output projection
    # ----------------------------------------------

    logits = Y @ W_out + b_out

    # ----------------------------------------------
    # Loss
    # ----------------------------------------------

    loss = F.cross_entropy(
        logits,
        targets
    )

    # ----------------------------------------------
    # Backpropagation
    # ----------------------------------------------

    loss.backward()

    # ----------------------------------------------
    # Update
    # ----------------------------------------------

    with torch.no_grad():

        for parameter in [
    embedding,
    position_embedding,
    W_Q,
    W_K,
    W_V,
    W1,
    b1,
    W2,
    b2,
    gamma1,
    beta1,
    gamma2,
    beta2,
    W_out,
    b_out
]:

            parameter -= learning_rate * parameter.grad

    # ----------------------------------------------
    # Clear gradients
    # ----------------------------------------------

    for parameter in [
    embedding,
    position_embedding,
    W_Q,
    W_K,
    W_V,
    W1,
    b1,
    W2,
    b2,
    gamma1,
    beta1,
    gamma2,
    beta2,
    W_out,
    b_out
]:

        parameter.grad.zero_()

    # ----------------------------------------------
    # Progress
    # ------for ----------------------------------------

    if step % 300 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f}"
        )


# ==================================================
# Final predictions
# ==================================================

predictions = torch.argmax(
    logits,
    dim=-1
)

print("\nInputs:")
print(inputs)

print("\nTargets:")
print(targets)

print("\nPredictions:")
print(predictions)

print("\nFinal attention:")
print(attention)
import torch

# 4 tokens, 3-dimensional embeddings
X = torch.tensor([
    [1.0, 0.0, 1.0],   # the
    [0.0, 1.0, 1.0],   # dog
    [1.0, 1.0, 0.0],   # chased
    [0.0, 1.0, 0.0]    # cat
])

W_Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

W_K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

W_V = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

# -------------------------
# Q, K, V
# -------------------------

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# -------------------------
# Attention scores
# -------------------------

scores = Q @ K.T
scores = scores / (K.shape[-1] ** 0.5)

# -------------------------
# Causal mask
# -------------------------

mask = torch.triu(
    torch.ones(4, 4),
    diagonal=1
)

scores = scores.masked_fill(mask == 1, float("-inf"))

# -------------------------
# Attention
# -------------------------

attention = torch.softmax(scores, dim=-1)

output = attention @ V

print("Masked scores:")
print(scores)

print("\nCausal attention weights:")
print(attention)

print("\nAttention output:")
print(output)
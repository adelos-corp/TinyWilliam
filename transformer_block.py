import torch

torch.manual_seed(42)

X = torch.randn(4, 8)

# A placeholder attention output.
# We'll replace this with our actual attention mechanism.
H = torch.randn(4, 8)

R = X + H

# Layer normalization

gamma = torch.ones(8)
beta = torch.zeros(8)

epsilon = 1e-5

mean = R.mean(dim=-1, keepdim=True)
variance = ((R - mean) ** 2).mean(dim=-1, keepdim=True)

R_normalized = (R - mean) / torch.sqrt(variance + epsilon)

Z = gamma * R_normalized + beta

# Feed-forward network

W1 = torch.randn(8, 32) * 0.1
b1 = torch.zeros(32)

W2 = torch.randn(32, 8) * 0.1
b2 = torch.zeros(8)

hidden = Z @ W1 + b1
activated = torch.nn.functional.gelu(hidden)
F = activated @ W2 + b2

# Second residual connection

Y = Z + F

# Second LayerNorm

gamma2 = torch.ones(8)
beta2 = torch.zeros(8)

mean2 = Y.mean(dim=-1, keepdim=True)
variance2 = ((Y - mean2) ** 2).mean(dim=-1, keepdim=True)

output = gamma2 * (
    (Y - mean2) / torch.sqrt(variance2 + epsilon)
) + beta2

print("\nFinal Transformer block output:")
print(output)

print("\nFinal means:")
print(output.mean(dim=-1))

print("\nFinal variances:")
print(output.var(dim=-1, unbiased=False))

print("\nFFN hidden shape:")
print(hidden.shape)

print("\nFFN output shape:")
print(F.shape)

print("\nMean:")
print(mean)

print("\nVariance:")
print(variance)

print("\nLayerNorm output:")
print(Z)

print("\nOutput means:")
print(Z.mean(dim=-1))

print("\nOutput variances:")
print(Z.var(dim=-1, unbiased=False))

print("X:")
print(X)

print("\nAttention output H:")
print(H)

print("\nResidual output X + H:")
print(R)
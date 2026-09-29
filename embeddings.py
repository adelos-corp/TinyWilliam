import torch

# Vocabulary
vocab = {
    "cat": 0,
    "dog": 1,
    "apple": 2,
    "car": 3,
    "banana": 4
}

# Embedding matrix
embedding = torch.randn(5, 3)

print("Embedding matrix:")
print(embedding)

# Some token IDs
tokens = torch.tensor([0, 1, 3])

# Look up their vectors
vectors = embedding[tokens]

print("\nToken IDs:")
print(tokens)

print("\nEmbedding vectors:")
print(vectors)
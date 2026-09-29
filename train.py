import torch
import torch.nn.functional as F

from dataset import inputs, targets
from tokenizer import stoi, encode, itos
from model import TinyWilliam

vocab_size = len(stoi)

# -----------------------------
# Model
# -----------------------------

model = TinyWilliam(
    vocab_size=vocab_size,
    context_length=32,
    d_model=24,
    d_ff=128
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# Training
# -----------------------------

steps = 5000

for step in range(steps):

    # Forward pass
    logits, attention = model(inputs)

    # Cross-entropy
    loss = F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1)
    )

    # Backward pass
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # Print progress
    if step % 500 == 0:
        print(
            f"Step {step:4d} | "
            f"Loss {loss.item():.6f}"
        )

print("\nProbabilities for position 3:")

probs = torch.softmax(logits, dim=-1)

for i in range(len(inputs)):
    print(f"\nExample {i}:")
    print("Input:", inputs[i])
    print("Target:", targets[i, 3].item())

    for token_id in range(vocab_size):
        print(
            token_id,
            probs[i, 3, token_id].item()
        )

# -----------------------------
# Final predictions
# -----------------------------

with torch.no_grad():

    logits, attention = model(inputs)

    predictions = logits.argmax(dim=-1)

print("\nPredictions:")
print(predictions)

print("\nAttention at position 3:")

for i in range(len(inputs)):
    print(
        f"Example {i}:",
        attention[i, 3]
    )

print("\nTargets:")
print(targets)

# TEST: An unseen sentence


test_sentence = "the fox eats"

# Encode and remove EOS because the sentence isn't finished
test_tokens = encode(test_sentence)[:-1]

# Convert to a batch of one sentence
test_input = torch.tensor([test_tokens])

model.eval()

with torch.no_grad():
    logits, attention = model(test_input)

    # Only the last position predicts the next word
    next_token_logits = logits[0, -1]

    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Find the 5 most probable next tokens
    top_probs, top_ids = torch.topk(probabilities, 5)

print("\nTINY WILLIAM'S FIRST EXAM")
print("Input:", test_sentence)

for probability, token_id in zip(top_probs, top_ids):
    print(
        itos[token_id.item()],
        f"{probability.item() * 100:.2f}%"
    )
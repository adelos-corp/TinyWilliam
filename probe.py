import torch

from tokenizer import encode, itos


# -----------------------------
# Train the model
# -----------------------------
#
# Importing train.py executes its training loop and leaves the
# trained model available. This keeps the probing script simple
# while the project is still in its experimental stage.
#

from train import model


# -----------------------------
# Probe final hidden states
# -----------------------------

model.eval()

test_sentences = [
    "the cat eats",
    "the dog eats",
    "the fox eats",
    "the bird eats",
    "the fox likes",
]

hidden_states = {}

print("\nREPRESENTATION PROBE")

with torch.no_grad():
    for sentence in test_sentences:
        tokens = encode(sentence)[:-1]
        x = torch.tensor([tokens])

        logits, attention = model(x)

        hidden = model.last_hidden[0, -1]
        hidden_states[sentence] = hidden

        print(f"\n{sentence}")
        print("Hidden vector:")
        print(hidden)

        probabilities = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_ids = torch.topk(probabilities, 5)

        print("Top predictions:")
        for probability, token_id in zip(top_probs, top_ids):
            print(
                f"  {itos[token_id.item()]:<10} "
                f"{probability.item() * 100:6.2f}%"
            )


# -----------------------------
# Compare representations
# -----------------------------
#
# Euclidean distance measures the absolute distance between
# two hidden vectors.
#
# Cosine similarity measures how closely their directions
# align, independent of their overall magnitude.
#

pairs = [
    ("the dog eats", "the fox eats"),
    ("the cat eats", "the dog eats"),
    ("the fox eats", "the fox likes"),
    ("the cat eats", "the bird eats"),
]

print("\nREPRESENTATION DISTANCES")

for first, second in pairs:
    a = hidden_states[first]
    b = hidden_states[second]

    euclidean = torch.linalg.vector_norm(a - b)
    cosine = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=1,
    ).item()

    print(f"\n{first}  <->  {second}")
    print(f"Euclidean distance: {euclidean.item():.4f}")
    print(f"Cosine similarity:  {cosine:.4f}")

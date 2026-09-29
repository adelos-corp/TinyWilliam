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
    "the dog likes",
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
# We compare contexts that differ in one important way:
#
# 1. Same verb, different animals.
# 2. Same animal, different verbs.
# 3. Different animals that lead to the same target.
# 4. Different animals that lead to different targets.
#
# Euclidean distance measures absolute separation.
# Cosine similarity measures directional alignment.
#

pairs = [
    ("the dog eats", "the fox eats"),
    ("the dog likes", "the fox likes"),
    ("the dog eats", "the dog likes"),
    ("the fox eats", "the fox likes"),
    ("the cat eats", "the dog eats"),
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


# -----------------------------
# Representation interpretation
# -----------------------------
#
# This does not assign a "meaning" to individual dimensions.
# Instead, it gives us controlled geometric comparisons.
#
# In particular, the most informative comparisons are:
#
# dog eats  <-> fox eats
# dog likes <-> fox likes
#
# Both pairs share the same predicted object respectively:
#
# meat  and  bones
#
# while:
#
# dog eats <-> dog likes
#
# keeps the animal fixed but changes the verb and target.

import torch

from model import TinyWilliam
from tokenizer import encode, itos, stoi


# -----------------------------
# Model configuration
# -----------------------------

model = TinyWilliam(
    vocab_size=len(stoi),
    context_length=32,
    d_model=24,
    d_ff=128
)


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
]

print("\nREPRESENTATION PROBE")

with torch.no_grad():
    for sentence in test_sentences:
        tokens = encode(sentence)[:-1]
        x = torch.tensor([tokens])

        logits, attention = model(x)

        hidden = model.last_hidden[0, -1]

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

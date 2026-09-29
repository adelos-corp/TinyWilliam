import re


# -----------------------------
# Training corpus
# -----------------------------

sentences = [
    "the cat eats fish",
    "the dog eats meat",
    "the cat likes milk",
    "the dog likes bones",
    "the bird eats seeds",
    "the bird likes worms",
    "the fox likes bones",
]


# -----------------------------
# Special tokens
# -----------------------------

special_tokens = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>"
]


# -----------------------------
# Build vocabulary
# -----------------------------

words = set()

for sentence in sentences:

    tokens = re.findall(
        r"\b\w+\b",
        sentence.lower()
    )

    words.update(tokens)


vocab_words = sorted(words)

itos = special_tokens + vocab_words

stoi = {
    token: i
    for i, token in enumerate(itos)
}


# -----------------------------
# Encode
# -----------------------------

def encode(text):

    tokens = re.findall(
        r"\b\w+\b",
        text.lower()
    )

    ids = [stoi["<BOS>"]]

    for token in tokens:

        ids.append(
            stoi.get(
                token,
                stoi["<UNK>"]
            )
        )

    ids.append(stoi["<EOS>"])

    return ids


# -----------------------------
# Decode
# -----------------------------

def decode(ids):

    tokens = []

    for token_id in ids:

        token = itos[token_id]

        if token in [
            "<BOS>",
            "<EOS>",
            "<PAD>"
        ]:
            continue

        tokens.append(token)

    return " ".join(tokens)


# -----------------------------
# Test
# -----------------------------

text = "the cat eats fish"

encoded = encode(text)

print("Vocabulary:")
print(stoi)

print("\nVocabulary size:")
print(len(stoi))

print("\nText:")
print(text)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(decode(encoded))
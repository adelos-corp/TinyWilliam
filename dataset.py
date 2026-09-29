import torch

from tokenizer import encode, sentences


encoded_sentences = [
    encode(sentence)
    for sentence in sentences
]


inputs = []
targets = []


for ids in encoded_sentences:

    inputs.append(ids[:-1])
    targets.append(ids[1:])


inputs = torch.tensor(inputs)
targets = torch.tensor(targets)


print("Encoded sentences:")

for sentence, ids in zip(
    sentences,
    encoded_sentences
):
    print(sentence, "->", ids)


print("\nInputs:")
print(inputs)

print("\nTargets:")
print(targets)

print("\nShapes:")
print("Inputs:", inputs.shape)
print("Targets:", targets.shape)
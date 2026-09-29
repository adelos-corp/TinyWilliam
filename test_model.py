import torch

from dataset import inputs, targets
from model import TinyWilliam


model = TinyWilliam(
    vocab_size=11,
    context_length=32,
    d_model=24,
    d_ff=128
)

logits = model(inputs)

print("Input shape:")
print(inputs.shape)

print("\nLogits shape:")
print(logits.shape)

print("\nExpected:")
print("(8, 4, 11)")

print("\nParameter count:")

total = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(total)
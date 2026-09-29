import torch
import torch.nn.functional as F


class TinyWilliam(torch.nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length=32,
        d_model=24,
        d_ff=128
    ):
        super().__init__()

        self.d_model = d_model
        self.context_length = context_length

        # Token embeddings
        self.token_embedding = torch.nn.Parameter(
            torch.randn(vocab_size, d_model) * 0.02
        )

        # Positional embeddings
        self.position_embedding = torch.nn.Parameter(
            torch.randn(context_length, d_model) * 0.02
        )

        # Attention
        self.WQ = torch.nn.Parameter(
            torch.randn(d_model, d_model) * 0.02
        )

        self.WK = torch.nn.Parameter(
            torch.randn(d_model, d_model) * 0.02
        )

        self.WV = torch.nn.Parameter(
            torch.randn(d_model, d_model) * 0.02
        )

        self.WO = torch.nn.Parameter(
            torch.randn(d_model, d_model) * 0.02
        )

        # LayerNorm 1
        self.gamma1 = torch.nn.Parameter(torch.ones(d_model))
        self.beta1 = torch.nn.Parameter(torch.zeros(d_model))

        # Feed-forward network
        self.W1 = torch.nn.Parameter(
            torch.randn(d_model, d_ff) * 0.02
        )

        self.b1 = torch.nn.Parameter(
            torch.zeros(d_ff)
        )

        self.W2 = torch.nn.Parameter(
            torch.randn(d_ff, d_model) * 0.02
        )

        self.b2 = torch.nn.Parameter(
            torch.zeros(d_model)
        )

        # LayerNorm 2
        self.gamma2 = torch.nn.Parameter(torch.ones(d_model))
        self.beta2 = torch.nn.Parameter(torch.zeros(d_model))

        # Output head
        self.W_out = torch.nn.Parameter(
            torch.randn(d_model, vocab_size) * 0.02
        )

    def layer_norm(self, x, gamma, beta):
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        normalized = (
            (x - mean)
            / torch.sqrt(variance + 1e-5)
        )

        return gamma * normalized + beta

    def forward(self, tokens):

        batch_size, sequence_length = tokens.shape

        # ------------------------------------------------
        # 1. Embeddings
        # ------------------------------------------------

        X = self.token_embedding[tokens]

        positions = torch.arange(
            sequence_length,
            device=tokens.device
        )

        X = X + self.position_embedding[positions]

        # ------------------------------------------------
        # 2. Self-Attention
        # ------------------------------------------------

        Q = X @ self.WQ
        K = X @ self.WK
        V = X @ self.WV

        scores = (
            Q @ K.transpose(-2, -1)
        ) / (self.d_model ** 0.5)

        # Causal mask
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=tokens.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask == 1,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention_output = attention @ V

        attention_output = attention_output @ self.WO

        # ------------------------------------------------
        # 3. Residual + LayerNorm
        # ------------------------------------------------

        X = X + attention_output

        X = self.layer_norm(
            X,
            self.gamma1,
            self.beta1
        )

        # ------------------------------------------------
        # 4. Feed Forward Network
        # ------------------------------------------------

        hidden = X @ self.W1 + self.b1

        hidden = F.gelu(hidden)

        FFN = hidden @ self.W2 + self.b2

        # ------------------------------------------------
        # 5. Residual + LayerNorm
        # ------------------------------------------------

        X = X + FFN

        X = self.layer_norm(
            X,
            self.gamma2,
            self.beta2
        )

        # ------------------------------------------------
        # 6. Save final hidden representation
        # ------------------------------------------------

        # Exposed for experiments and representation probing.
        self.last_hidden = X

        # ------------------------------------------------
        # 7. Output logits
        # ------------------------------------------------

        logits = X @ self.W_out

        return logits, attention
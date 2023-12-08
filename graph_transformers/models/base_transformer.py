import torch
import torch.nn as nn
from typing import Tuple, Union, Optional, List

# import torchvision
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionHead(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        """
        Args:
            dim: The input and output dimension of the attention head
            n_hidden: The hidden dimension of the attention head
        """
        super().__init__()
        self.W_K = nn.Linear(dim, n_hidden)
        self.W_Q = nn.Linear(dim, n_hidden)
        self.W_V = nn.Linear(dim, n_hidden)
        self.n_hidden = n_hidden

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            attn_mask: Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            out: Output tensor of shape (batch_size, seq_len, dim)
            alpha: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        # Compute Q, K, V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.n_hidden, dtype=torch.float32)
        )

        # Apply the attention mask (if provided)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        # Compute attention weights (alpha)
        alpha = F.softmax(attn_scores, dim=-1)

        # Compute the output as a weighted sum of the values
        out = torch.matmul(alpha, V)

        return out, alpha


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim: int, n_hidden: int, num_heads: int):
        """
        Args:
            dim: The input and output dimension of the attention head
            n_hidden: The hidden dimension of the attention head
            num_heads: The number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.n_hidden = n_hidden // num_heads

        # Ensure the hidden dimension is divisible by the number of heads
        assert (
            self.n_hidden * num_heads == n_hidden
        ), "n_hidden must be divisible by num_heads"

        self.heads = nn.ModuleList(
            [AttentionHead(dim, self.n_hidden) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(n_hidden, dim)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            attn_mask: Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            attn_output: Output tensor of shape (batch_size, seq_len, dim)
            attn_alphas: Attention weights of shape (num_heads, batch_size, seq_len, seq_len)
        """
        head_outputs, alphas = [], []

        for head in self.heads:
            head_output, alpha = head(x, attn_mask)
            head_outputs.append(head_output)
            alphas.append(alpha.unsqueeze(1))

        # Concatenate the outputs and pass through the final linear layer
        concat_output = torch.cat(head_outputs, dim=-1)
        attn_output = self.linear(concat_output)

        # Stack the alphas for each head
        attn_alphas = torch.cat(alphas, dim=1)

        return attn_output, attn_alphas


class FFN(nn.Module):
    def __init__(self, dim: int, n_hidden: int):
        """
        Args:
            dim: The input and output dimension of the FFN
            n_hidden: The hidden dimension of the FFN
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionResidual(nn.Module):
    def __init__(self, dim: int, attn_dim: int, mlp_dim: int, num_heads: int):
        """
        Args:
            dim: The input and output dimension of the attention head
            attn_dim: The hidden dimension of the attention head
            mlp_dim: The hidden dimension of the FFN
            num_heads: The number of attention heads
        """
        super().__init__()
        self.attn = MultiHeadedAttention(dim, attn_dim, num_heads)
        self.ffn = FFN(dim, mlp_dim)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            attn_mask: Attention mask of shape (batch_size, seq_len, seq_len)

        Returns:
            x: Output tensor of shape (batch_size, seq_len, dim)
            alphas: Attention weights of shape (num_heads, batch_size, seq_len, seq_len)
        """

        attn_out, alphas = self.attn(x=x, attn_mask=attn_mask)
        x = attn_out + x
        x = self.ffn(x) + x
        return x, alphas


class OutLayer(nn.Module):
    def __init__(self, dim: int, seq_len: int | None, out_dim: int):
        if seq_len is None:
            raise ValueError("seq_len must be provided if out_dim is not None")

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(dim * seq_len, out_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        return self.net(x), None


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_dim: int,
        mlp_dim: int,
        num_heads: int,
        num_layers: int,
        seq_len: int | None = None,
        out_dim=None,
    ):
        """
        Args:
            dim: The input and output dimension of the attention head
            attn_dim: The hidden dimension of the attention head
            mlp_dim: The hidden dimension of the FFN
            num_heads: The number of attention heads
            num_layers: The number of transformer layers
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AttentionResidual(dim, attn_dim, mlp_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        if out_dim is not None:
            self.layers.append(OutLayer(dim, seq_len, out_dim))

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor, return_attn=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            attn_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            return_attn: Whether to return the attention weights

        Returns:
            output: Output tensor of shape (batch_size, seq_len, dim)
            collected_attns: Attention weights of shape (batch_size, num_heads, num_tokens, num_tokens)
        """
        collected_attns = []

        for layer in self.layers:
            x, alphas = layer(x, attn_mask)
            if return_attn:
                if alphas is not None:
                    # Permute alphas to shape (batch_size, num_heads, num_tokens, num_tokens)
                    # alphas = alphas.permute(1, 0, 2, 3)
                    collected_attns.append(alphas.unsqueeze(1))

        output = x
        collected_attns = torch.cat(collected_attns, dim=1) if return_attn else None

        return output, collected_attns

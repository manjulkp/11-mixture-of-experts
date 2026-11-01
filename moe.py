"""A toy mixture of experts (MoE) in the Tiny GPT.

A dense model runs its one large feed-forward layer for every token. A MoE model
holds several smaller feed-forward layers (experts), and a small router sends each
token to only a few of them. The model can then hold many parameters, while each
token only uses some of them.
"""

import torch
from torch import nn
from torch.nn import functional as F

from tinygpt.model import GPT, GPTConfig


class SizedFeedForward(nn.Module):
    def __init__(self, n_embd, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, n_embd, n_experts, top_k, expert_hidden, dropout=0.0):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be between 1 and n_experts")
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(n_embd, n_experts, bias=False)
        self.experts = nn.ModuleList(SizedFeedForward(n_embd, expert_hidden, dropout) for _ in range(n_experts))
        self.balance_loss = torch.zeros(())
        self.register_buffer("usage", torch.zeros(n_experts), persistent=False)

    def forward(self, x):
        router_logits = self.router(x)
        top_values, top_indices = router_logits.topk(self.top_k, dim=-1)
        gates = F.softmax(top_values, dim=-1)

        output = torch.zeros_like(x)
        for expert_index, expert in enumerate(self.experts):
            # Find the tokens routed here, and run this expert on only those.
            batch, position, slot = (top_indices == expert_index).nonzero(as_tuple=True)
            if len(batch):
                result = expert(x[batch, position])
                output[batch, position] += gates[batch, position, slot, None] * result

        # Fraction of routing slots each expert received, for the balance loss and reports.
        counts = torch.bincount(top_indices.flatten(), minlength=self.n_experts).float()
        fractions = counts / counts.sum()
        if not self.training:
            self.usage += counts
        # Switch Transformer's load-balancing loss: it's lowest when both the actual
        # share and the router's average probability are even across experts.
        mean_probability = F.softmax(router_logits, dim=-1).mean(dim=(0, 1))
        self.balance_loss = self.n_experts * (fractions.detach() * mean_probability).sum()
        return output


def replace_feed_forward(model, make_layer):
    for block in model.blocks:
        block.feed_forward = make_layer()
    model.apply(GPT._init_weights)
    return model


class MoEGPT(GPT):
    """The Tiny GPT with its feed-forward layers swapped for MoE layers."""

    def __init__(self, config, n_experts=4, top_k=2, expert_hidden=None, balance_weight=0.01):
        super().__init__(config)
        expert_hidden = expert_hidden or config.n_embd
        self.balance_weight = balance_weight
        replace_feed_forward(
            self, lambda: MoELayer(config.n_embd, n_experts, top_k, expert_hidden, config.dropout)
        )

    def forward(self, ids, targets=None):
        logits, loss = super().forward(ids, targets)
        if loss is not None and self.training and self.balance_weight:
            # The balance term only applies during training, so that validation
            # loss stays comparable with the dense models.
            balance = sum(block.feed_forward.balance_loss for block in self.blocks) / len(self.blocks)
            loss = loss + self.balance_weight * balance
        return logits, loss

    def usage_shares(self):
        """Return each layer's share of routed tokens per expert, since the last reset."""
        return [
            (block.feed_forward.usage / block.feed_forward.usage.sum().clamp_min(1)).tolist()
            for block in self.blocks
        ]

    def reset_usage(self):
        for block in self.blocks:
            block.feed_forward.usage.zero_()


def build_dense(config, hidden_multiplier):
    """Build a GPT whose feed-forward layers use a chosen hidden size."""
    model = GPT(config)
    hidden = int(config.n_embd * hidden_multiplier)
    return replace_feed_forward(model, lambda: SizedFeedForward(config.n_embd, hidden, config.dropout))


def active_parameters_per_token(n_embd, hidden, top_k=1, router=0):
    """Count the feed-forward parameters that one token actually uses, per layer."""
    one = (n_embd + 1) * hidden + (hidden + 1) * n_embd
    return router + top_k * one


__all__ = ["GPTConfig", "MoEGPT", "MoELayer", "active_parameters_per_token", "build_dense"]

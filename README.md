# Project 11: Toy mixture of experts (stretch)

A dense model runs its one large feed-forward layer for every token. A mixture of
experts (MoE) holds several smaller layers, and a small router sends each token to only
a few of them. The model can then hold many parameters while each token uses only some.

| Model | Compared on |
| --- | --- |
| Dense, 4x hidden size | About the same total parameters as the MoE |
| Dense, 2x hidden size | About the same per-token cost as the MoE |
| MoE, 4 experts, top 2, with balancing | A loss term encourages even traffic across experts |
| MoE, 4 experts, top 2, no balancing | Shows what routing does without that loss term |

## Results

**Not available yet.** The experiment hasn't finished. When it has, results will be in
`outputs\results.md` and `outputs\comparison.png`, including each expert's share of
traffic.

## Run it

This uses the data from Project 4, so run its `prepare_data.py` first.

```powershell
uv sync
uv run python experiment.py
uv run pytest
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--iters` | `2000` | Training iterations per run |
| `--seeds` | `2` | Seeds per model |
| `--layers` | `3` | Transformer blocks |
| `--embd` | `96` | Embedding size |
| `--block` | `64` | Longest context, in characters |

## Files

| File | What it does |
| --- | --- |
| `moe.py` | The router, experts and load-balancing loss |
| `experiment.py` | Trains all four models and measures expert usage |
| `tests/` | Includes a check that unchosen experts get no gradient |

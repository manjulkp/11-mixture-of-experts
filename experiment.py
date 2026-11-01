"""Compare dense feed-forward layers with a mixture of experts on the same data."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from moe import MoEGPT, active_parameters_per_token, build_dense
from tinygpt.data import get_batch, load_splits
from tinygpt.model import GPTConfig
from tinygpt.plots import INK_SECONDARY, SERIES, plt, style_axes
from tinygpt.training import TrainConfig, train


def make_variants(config):
    n = config.n_embd
    return {
        "Dense, 4x hidden size (same total parameters as MoE)": {
            "build": lambda: build_dense(config, 4),
            "active": active_parameters_per_token(n, 4 * n),
        },
        "Dense, 2x hidden size (same per-token cost as MoE)": {
            "build": lambda: build_dense(config, 2),
            "active": active_parameters_per_token(n, 2 * n),
        },
        "MoE, 4 experts, top 2, with balancing": {
            "build": lambda: MoEGPT(config, 4, 2, n, balance_weight=0.01),
            "active": active_parameters_per_token(n, n, top_k=2, router=4 * n),
        },
        "MoE, 4 experts, top 2, no balancing": {
            "build": lambda: MoEGPT(config, 4, 2, n, balance_weight=0.0),
            "active": active_parameters_per_token(n, n, top_k=2, router=4 * n),
        },
    }


def evenness(shares):
    """Normalized entropy: 1 means all experts are used equally, 0 means only one is."""
    entropy = -sum(share * math.log(share) for share in shares if share > 0)
    return entropy / math.log(len(shares))


@torch.no_grad()
def measure_usage(model, val_ids, block_size, batches=20):
    model.eval()
    model.reset_usage()
    generator = torch.Generator().manual_seed(0)
    for _ in range(batches):
        model(get_batch(val_ids, block_size, 32, generator)[0])
    return model.usage_shares()


def plot(rows, path):
    rows = sorted(rows, key=lambda row: -row["val_mean"])
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.5))
    ax.hlines(y, [r["val_min"] for r in rows], [r["val_max"] for r in rows], color=SERIES[0], lw=1.6, alpha=0.45)
    ax.scatter([r["val_mean"] for r in rows], y, color=SERIES[0], s=46, edgecolors="#fcfcfb", linewidths=1.5, zorder=3)
    ax.set_yticks(list(y), [r["name"] for r in rows])
    style_axes(ax, grid="x")
    ax.margins(x=0.2, y=0.2)
    ax.set_xlabel("Best validation loss (lower is better)", color=INK_SECONDARY)
    ax.set_title("Dense vs mixture of experts: dot is the mean, line is the range across seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="../04-tiny-gpt/data")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--embd", type=int, default=96)
    parser.add_argument("--block", type=int, default=64)
    args = parser.parse_args(argv)

    if not (Path(args.data) / "train.txt").exists():
        parser.error(f"{args.data}/train.txt not found. Run prepare_data.py in Project 4 first.")

    tokenizer, train_ids, val_ids = load_splits(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = GPTConfig(tokenizer.vocab_size, args.block, args.layers, args.heads, args.embd, dropout=0.0)

    rows = []
    for name, variant in make_variants(config).items():
        losses, seconds, usage = [], [], None
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = variant["build"]()
            train_config = TrainConfig(max_iters=args.iters, seed=seed, eval_every=max(1, args.iters // 8), eval_iters=30)
            history = train(model, train_ids, val_ids, train_config, log=None)
            losses.append(min(row["val_loss"] for row in history))
            seconds.append(history[-1]["seconds"])
            print(f"{name:<55} seed {seed}  val {losses[-1]:.4f}  {seconds[-1]:.0f}s", flush=True)
            if isinstance(model, MoEGPT):
                usage = measure_usage(model, val_ids, args.block)
        rows.append(
            {
                "name": name,
                "parameters": model.num_parameters(),
                "active_per_layer": variant["active"],
                "val_mean": statistics.mean(losses),
                "val_min": min(losses),
                "val_max": max(losses),
                "seconds": statistics.mean(seconds),
                "usage": usage,
                "evenness": statistics.mean(evenness(layer) for layer in usage) if usage else None,
            }
        )

    lines = [
        "# Mixture of experts results",
        "",
        f"{args.layers} layers, {args.embd} embedding size, {args.iters} iterations, {args.seeds} seeds. "
        "Active parameters counts what one token uses in one layer's feed-forward step. "
        "Evenness is 1 when all experts get equal traffic and 0 when one gets everything.",
        "",
        "| Model | Total parameters | Active parameters | Best validation loss | Seed range | Seconds per run | Expert evenness |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: r["val_mean"]):
        even = "Not applicable" if row["evenness"] is None else f"{row['evenness']:.3f}"
        lines.append(
            f"| {row['name']} | {row['parameters']:,} | {row['active_per_layer']:,} | {row['val_mean']:.4f} | "
            f"{row['val_min']:.4f} to {row['val_max']:.4f} | {row['seconds']:.0f} | {even} |"
        )
    lines += ["", "## Share of traffic per expert (last seed)", ""]
    for row in rows:
        if row["usage"]:
            lines.append(f"**{row['name']}**")
            lines.append("")
            lines += [
                f"- Layer {i + 1}: " + ", ".join(f"{share:.0%}" for share in layer)
                for i, layer in enumerate(row["usage"])
            ]
            lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    plot(rows, out / "comparison.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

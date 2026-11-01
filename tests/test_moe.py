import pytest
import torch

from experiment import evenness, main, make_variants, measure_usage
from moe import MoEGPT, MoELayer, active_parameters_per_token, build_dense
from tinygpt.model import GPT, GPTConfig

CONFIG = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def layer(top_k=2):
    torch.manual_seed(0)
    return MoELayer(16, 4, top_k, 16)


def test_output_shape_matches_input():
    x = torch.randn(3, 5, 16)
    assert layer()(x).shape == x.shape


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        MoELayer(8, 2, 3, 8)


def test_output_is_the_gated_sum_of_only_the_chosen_experts():
    moe = layer().eval()
    x = torch.randn(1, 1, 16)
    with torch.no_grad():
        output = moe(x)
        values, indices = moe.router(x).topk(2, dim=-1)
        gates = values.softmax(-1)[0, 0]
        chosen = indices[0, 0].tolist()
        expected = gates[0] * moe.experts[chosen[0]](x[0, 0]) + gates[1] * moe.experts[chosen[1]](x[0, 0])
    assert torch.allclose(output[0, 0], expected, atol=1e-6)


def test_unchosen_experts_get_no_gradient():
    moe = layer(top_k=1)
    x = torch.randn(1, 1, 16)
    moe(x).sum().backward()
    with_gradient = sum(e.net[0].weight.grad is not None for e in moe.experts)
    assert with_gradient == 1


def test_router_receives_gradient_through_the_gates():
    moe = layer()
    moe(torch.randn(2, 4, 16)).sum().backward()
    assert moe.router.weight.grad.abs().sum() > 0


def test_usage_counts_only_in_eval_mode():
    moe = layer()
    moe.train()
    moe(torch.randn(1, 5, 16))
    assert moe.usage.sum() == 0
    moe.eval()
    moe(torch.randn(1, 5, 16))
    assert moe.usage.sum() == 10  # 5 tokens with 2 experts each


def test_balance_loss_is_one_when_balanced_and_higher_when_not():
    moe = MoELayer(4, 2, 1, 4)
    # Half the tokens choose each expert, with symmetric probabilities.
    with torch.no_grad():
        moe.router.weight.copy_(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    balanced = torch.tensor([[[9.0, 0, 0, 0], [0, 9.0, 0, 0]]])
    moe(balanced)
    assert moe.balance_loss.item() == pytest.approx(1.0, abs=1e-3)
    moe(torch.tensor([[[9.0, 0, 0, 0], [9.0, 0, 0, 0]]]))
    assert moe.balance_loss.item() == pytest.approx(2.0, abs=1e-3)


def test_moe_model_is_causal():
    torch.manual_seed(0)
    model = MoEGPT(CONFIG).eval()
    ids = torch.randint(20, (1, 12))
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % 20
    with torch.no_grad():
        assert torch.allclose(model(ids)[0][0, :8], model(changed)[0][0, :8], atol=1e-6)


def test_balance_term_applies_only_during_training():
    torch.manual_seed(0)
    model = MoEGPT(CONFIG, balance_weight=1.0)
    ids = torch.randint(20, (2, 8))
    train_loss = model(ids, ids)[1]
    model.eval()
    eval_loss = model(ids, ids)[1]
    assert train_loss.item() == pytest.approx(eval_loss.item() + 1.0, abs=0.5)
    assert train_loss.item() > eval_loss.item() + 0.5


def test_dense_hidden_multiplier_changes_size():
    torch.manual_seed(0)
    assert build_dense(CONFIG, 4).num_parameters() > build_dense(CONFIG, 2).num_parameters()
    # A 4x dense layer has the same structure as the original model.
    assert build_dense(CONFIG, 4).num_parameters() == GPT(CONFIG).num_parameters()


def test_moe_total_parameters_close_to_dense_4x():
    moe = MoEGPT(CONFIG).num_parameters()
    dense = build_dense(CONFIG, 4).num_parameters()
    assert abs(moe - dense) / dense < 0.1


def test_active_parameter_count():
    # 2 inputs + bias to 3 hidden, and 3 hidden + bias to 2 outputs: 9 + 8.
    assert active_parameters_per_token(2, 3) == 17
    assert active_parameters_per_token(2, 3, top_k=2, router=8) == 42


def test_evenness():
    assert evenness([0.25] * 4) == pytest.approx(1.0)
    assert evenness([1.0, 0, 0, 0]) == pytest.approx(0.0)


def test_measure_usage_shares_sum_to_one():
    torch.manual_seed(0)
    shares = measure_usage(MoEGPT(CONFIG), torch.randint(20, (200,)), 16, batches=2)
    assert len(shares) == 2
    assert all(sum(layer_shares) == pytest.approx(1.0) for layer_shares in shares)


def test_all_variants_build_and_train_one_step():
    for variant in make_variants(CONFIG).values():
        model = variant["build"]()
        ids = torch.randint(20, (2, 8))
        model(ids, ids)[1].backward()


def test_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.txt").write_text("ANNA: hello there\nBEN: hi\n\n" * 50)
    (data / "val.txt").write_text("BEN: hello\nANNA: hi there\n\n" * 10)
    out = tmp_path / "out"
    main(["--data", str(data), "--out", str(out), "--iters", "4", "--seeds", "1", "--layers", "1", "--embd", "16", "--block", "16"])
    assert "Expert evenness" in (out / "results.md").read_text(encoding="utf-8")
    assert (out / "comparison.png").exists()

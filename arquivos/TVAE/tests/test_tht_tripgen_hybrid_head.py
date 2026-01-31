import pytest

torch = pytest.importorskip("torch")

from models.tht_tripgen import THTTripGenModel


def _build_hybrid_model() -> THTTripGenModel:
    num_zones = 5
    num_time_bins = 4
    emb_dim = 6
    cond_card = {"dow_idx": 3}
    zone_embeddings = torch.randn(num_zones, emb_dim)
    return THTTripGenModel(
        num_zones=num_zones,
        num_time_bins=num_time_bins,
        conditional_cardinalities=cond_card,
        frozen_zone_embeddings=zone_embeddings,
        cond_emb_dim=4,
        time_emb_dim=4,
        origin_emb_dim=5,
        context_mlp_hidden=12,
        context_mlp_layers=2,
        dropout=0.0,
        destination_head_type="hybrid",
        hybrid_residual_hidden=8,
        hybrid_residual_weight_init=1.0,
        use_passenger_count=False,
        passenger_cardinality=None,
        use_total_amount=False,
        total_amount_sigma_floor=1e-4,
        min_r_eps=1e-6,
        residual_num_layers=2,
        residual_num_bins=4,
        residual_context_hidden=8,
        residual_min_bin_width=1e-2,
        residual_min_bin_height=1e-2,
        residual_min_deriv=1e-2,
        residual_eps=1e-6,
    )


def test_hybrid_head_forward_and_frozen_embeddings():
    torch.manual_seed(0)
    model = _build_hybrid_model()

    batch = 6
    u = {"dow_idx": torch.randint(0, 3, (batch,))}
    h_idx = torch.randint(0, 4, (batch,))
    o_idx = torch.randint(0, 5, (batch,))
    d_idx = torch.randint(0, 5, (batch,))

    out = model(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx)
    assert out["logits_d"].shape == (batch, 5)
    assert model.zone_embeddings.requires_grad is False


def test_hybrid_head_train_step_runs():
    torch.manual_seed(1)
    model = _build_hybrid_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = 4
    u = {"dow_idx": torch.randint(0, 3, (batch,))}
    h_idx = torch.randint(0, 4, (batch,))
    o_idx = torch.randint(0, 5, (batch,))
    d_idx = torch.randint(0, 5, (batch,))
    r = torch.rand(batch)

    loss = model.nll(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx, r=r)["nll_total"]
    loss.backward()
    optimizer.step()

    assert model.dest_hybrid is not None
    residual_grad = model.dest_hybrid.residual_out.weight.grad
    assert residual_grad is not None

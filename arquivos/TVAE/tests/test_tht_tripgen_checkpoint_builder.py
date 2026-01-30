import pytest


torch = pytest.importorskip("torch")

from utils.serialization import (
    build_tht_tripgen_model_from_checkpoint,
    save_checkpoint,
    load_checkpoint,
)
from models.tht_tripgen import THTTripGenModel


def test_tht_tripgen_checkpoint_build_and_forward(tmp_path):
    num_zones = 4
    num_time_bins = 3
    emb_dim = 6
    cond_card = {"dow_idx": 3}

    zone_embeddings = torch.randn(num_zones, emb_dim)
    model = THTTripGenModel(
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
        destination_head_type="embedding_softmax",
        min_r_eps=1e-6,
        residual_num_layers=2,
        residual_num_bins=4,
        residual_context_hidden=8,
        residual_min_bin_width=1e-2,
        residual_min_bin_height=1e-2,
        residual_min_deriv=1e-2,
        residual_eps=1e-6,
    )

    meta = {
        "num_zones": num_zones,
        "num_time_bins": num_time_bins,
        "conditional_cardinalities": cond_card,
        "zone_embeddings_source": "Efunc",
        "zone_embeddings_dim": emb_dim,
        "cond_emb_dim": 4,
        "time_emb_dim": 4,
        "origin_emb_dim": 5,
        "context_mlp_hidden": 12,
        "context_mlp_layers": 2,
        "dropout": 0.0,
        "destination_head_type": "embedding_softmax",
        "min_r_eps": 1e-6,
        "residual_num_layers": 2,
        "residual_num_bins": 4,
        "residual_context_hidden": 8,
        "residual_min_bin_width": 1e-2,
        "residual_min_bin_height": 1e-2,
        "residual_min_deriv": 1e-2,
        "residual_eps": 1e-6,
    }

    ckpt_path = tmp_path / "tht_tripgen.pt"
    save_checkpoint(
        ckpt_path,
        model_state=model.state_dict(),
        meta=meta,
        optimizer_state=None,
        epoch=1,
        metrics=None,
    )

    state = load_checkpoint(ckpt_path)
    rebuilt = build_tht_tripgen_model_from_checkpoint(state)
    rebuilt.load_state_dict(state["model_state"])

    batch = 5
    u = {"dow_idx": torch.randint(0, cond_card["dow_idx"], (batch,))}
    h_idx = torch.randint(0, num_time_bins, (batch,))
    o_idx = torch.randint(0, num_zones, (batch,))
    d_idx = torch.randint(0, num_zones, (batch,))

    out = rebuilt(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx)
    assert out["logits_h"].shape == (batch, num_time_bins)
    assert out["logits_o"].shape == (batch, num_zones)
    assert out["logits_d"].shape == (batch, num_zones)

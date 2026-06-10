"""Smoke test for PhenoHER2 v2.

Runs:
    1. Forward + backward with random tensors at varying N (~1000..6000 patches).
    2. Loss returns finite parts: ce, emd, aux01, high3p, balance, orthogonality.
    3. predict() returns calibrated probs (sum to 1) and a pred_class.
    4. predict_with_tta() returns mean+std and is shape-consistent.
    5. fit_temperature() runs against a fake DataLoader and updates T to a
       finite positive scalar.
    6. Bag mixup helper produces a sane mixed bag and soft target that sums to 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]  # .../gradCode
sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from herohe.gp2.models import (
    PhenoHER2,
    PhenoHER2Config,
    PhenoHER2Loss,
    LossWeights,
)
from herohe.gp2.scripts.train_phenotype_mil import bag_mixup, soft_target_loss


def _fake_batch(B: int, N: int, D: int, K: int, num_classes: int = 4):
    x = torch.randn(B, N, D)
    coords = torch.randn(B, N, 2) * 100.0
    y = torch.randint(0, num_classes, (B,))
    return x, coords, y


def main():
    device = torch.device("cpu")  # smoke runs on CPU for portability
    torch.manual_seed(0)
    np.random.seed(0)

    cfg = PhenoHER2Config(
        feature_dim=2560,
        hidden_dim=64,
        num_prototypes=8,
        num_classes=4,
        attn_hidden_dim=32,
        cross_proto_layers=1,
        cross_proto_heads=2,
        dropout=0.1,
        use_spatial_block=True,
        spatial_block_heads=2,
        use_cls_pool=False,
        patch_dropout=0.2,
        use_dual_stream=True,
        detail_attn_hidden=32,
    )
    model = PhenoHER2(cfg).to(device)
    print(f"[smoke v2] params={sum(p.numel() for p in model.parameters()):,}")

    # Class counts: simulate HEROHE imbalance roughly
    class_counts = np.array([60, 90, 200, 50])
    loss_fn = PhenoHER2Loss(
        class_counts=class_counts,
        prototype_param=model.prototypes,
        weights=LossWeights(),
    ).to(device)

    # 1. forward + backward with two different N (training mode)
    model.train()
    for N in (256, 1500, 4096):
        x, c, y = _fake_batch(B=1, N=N, D=cfg.feature_dim, K=cfg.num_prototypes)
        x = x.to(device); c = c.to(device); y = y.to(device)
        out = model(x, coords=c)
        # Shape checks
        assert out["logits_4cls"].shape == (1, 4), out["logits_4cls"].shape
        assert out["logits_phen"].shape == (1, 4)
        assert out["logits_detail"].shape == (1, 4)
        assert out["fusion_gate"].shape == (1, 1)
        assert out["soft_assign"].shape[1:] == (out["soft_assign"].shape[1], cfg.num_prototypes)
        assert out["assign_sim"].shape == out["soft_assign"].shape
        # backward
        total, parts = loss_fn(out, y)
        assert torch.isfinite(total), f"non-finite loss for N={N}"
        for k, v in parts.items():
            assert torch.isfinite(v), f"non-finite part {k} for N={N}: {v}"
        total.backward()
        print(f"[smoke v2] train N={N:>5}  total={float(total.detach()):.4f}  "
              f"parts={ {k: round(float(v),4) for k,v in parts.items()} }  "
              f"gate={float(out['fusion_gate'].detach()):.3f}")
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

    # 2. eval mode + predict() + TTA
    model.eval()
    x, c, y = _fake_batch(B=1, N=2048, D=cfg.feature_dim, K=cfg.num_prototypes)
    x = x.to(device); c = c.to(device); y = y.to(device)
    pred = model.predict(x, coords=c, apply_calibration=True)
    probs = pred["probs"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, device=device), atol=1e-4)
    assert pred["pred_class"].shape == (1,)
    print(f"[smoke v2] predict probs={probs.detach().cpu().numpy().round(3).tolist()}  "
          f"pred={int(pred['pred_class'].item())}")

    tta = model.predict_with_tta(x, coords=c, n_samples=4, subsample_frac=0.7)
    assert tta["probs"].shape == (1, 4)
    assert tta["probs_std"].shape == (1, 4)
    print(f"[smoke v2] tta probs_mean={tta['probs'].detach().cpu().numpy().round(3).tolist()} "
          f"std={tta['probs_std'].detach().cpu().numpy().round(3).tolist()}")

    # 3. Fake val "loader" -> fit_temperature
    class FakeLoader:
        def __init__(self, n=8):
            self.batches = []
            for _ in range(n):
                xx, cc, yy = _fake_batch(1, 256, cfg.feature_dim, cfg.num_prototypes)
                self.batches.append({"features": xx, "coords": cc, "label": yy})
        def __iter__(self):
            return iter(self.batches)

    T = model.fit_temperature(FakeLoader(), device=device, max_iter=20)
    assert np.isfinite(T) and T > 0
    print(f"[smoke v2] fit_temperature T={T:.4f}  "
          f"calibration_temperature={float(model.calibration_temperature.item()):.4f}")

    # 4. Bag mixup helper
    rng = np.random.default_rng(0)
    xa, ca, ya = _fake_batch(1, 800, cfg.feature_dim, cfg.num_prototypes)
    xb, cb, yb = _fake_batch(1, 1200, cfg.feature_dim, cfg.num_prototypes)
    mx, mc, st = bag_mixup(
        feats_a=xa, feats_b=xb, coords_a=ca, coords_b=cb,
        y_a=int(ya.item()), y_b=int(yb.item()), num_classes=4,
        alpha=0.4, rng=rng, label_smoothing=0.1,
    )
    assert mx.shape[0] == 1 and mx.shape[2] == cfg.feature_dim
    assert mc is not None and mc.shape == (1, mx.shape[1], 2)
    assert torch.allclose(st.sum(), torch.tensor(1.0), atol=1e-4)
    out = model(mx, coords=mc)
    soft_loss = soft_target_loss(out, st, loss_fn.ce_weights)
    assert torch.isfinite(soft_loss)
    print(f"[smoke v2] mixup bag N={mx.shape[1]}  soft_loss={float(soft_loss):.4f}  "
          f"target={st.detach().cpu().numpy().round(3).tolist()}")

    # 5. Squared EMD: a perfect prediction yields ~0; far-off yields >0.
    from herohe.gp2.models.losses import squared_emd_loss, soft_ordinal_target
    p_perfect = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    t_2 = soft_ordinal_target(torch.tensor([2]), 4, 0.0)
    emd_close = squared_emd_loss(p_perfect, t_2)
    p_far = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    emd_far = squared_emd_loss(p_far, soft_ordinal_target(torch.tensor([3]), 4, 0.0))
    assert emd_close.item() < 1e-6
    assert emd_far.item() > 0.5
    print(f"[smoke v2] EMD^2 perfect={float(emd_close):.6f}  worst-ish={float(emd_far):.6f}")

    print("[smoke v2] all checks passed.")


if __name__ == "__main__":
    main()

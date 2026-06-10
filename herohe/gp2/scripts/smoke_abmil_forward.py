"""Smoke test: load a single Virchow2 feature bag and run AB-MIL forward+backward.

Usage:
    python smoke_abmil_forward.py \
        --features_h5 /path/to/3.h5 \
        [--label 0]    # optional, only needed to test backward()

Verifies:
    * h5 layout (features + coords) loads correctly.
    * AB-MIL forward pass produces sensible logits and an attention vector
      that sums to 1 over the patches.
    * One backward step works on the chosen device (mps / cuda / cpu).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# Allow `python smoke_abmil_forward.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from abmil import ABMIL, ABMILConfig, count_parameters  # noqa: E402


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_h5", type=Path, required=True,
                   help="Path to a TRIDENT-style feature .h5 with 'features' and 'coords'.")
    p.add_argument("--label", type=int, default=None,
                   help="Optional integer class label to also run a backward step.")
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--attn_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"[device] {device}")

    with h5py.File(args.features_h5, "r") as f:
        feats = f["features"][:]
        coords = f["coords"][:]
        encoder = f["features"].attrs.get("encoder", "?")
        slide_name = f["features"].attrs.get("name", args.features_h5.stem)

    n_patches, in_dim = feats.shape
    print(f"[bag] slide={slide_name} encoder={encoder} N={n_patches} D={in_dim} "
          f"coords_shape={coords.shape}")

    cfg = ABMILConfig(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    )
    model = ABMIL(cfg).to(device)
    print(f"[model] AB-MIL params={count_parameters(model):,} cfg={cfg}")

    x = torch.from_numpy(feats).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)
    logits = out["logits"]
    attn = out["attn"]
    probs = F.softmax(logits, dim=-1)

    print(f"[forward] logits={logits.detach().cpu().numpy().round(4)}")
    print(f"[forward] probs ={probs.detach().cpu().numpy().round(4)}")
    print(f"[forward] attn  shape={tuple(attn.shape)} sum={attn.sum().item():.4f} "
          f"max={attn.max().item():.4f} min={attn.min().item():.4f} "
          f"top5_idx={attn.topk(5).indices.detach().cpu().tolist()}")

    if args.label is not None:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad()
        out = model(x)
        target = torch.tensor([args.label], device=device)
        loss = F.cross_entropy(out["logits"].unsqueeze(0), target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"[backward] label={args.label} loss={loss.item():.6f} "
              f"grad_norm(pre-clip)={grad_norm.item():.4f}")

    print("[ok] smoke test passed")


if __name__ == "__main__":
    main()

"""Compare prototype separation: vector cosine, label correlation, spatial patch_attn overlap.

Writes JSON summary + optional PNG for one slide (201 by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import h5py

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device):
    blob = torch.load(checkpoint, map_location="cpu")
    names = {f.name for f in fields(PhenoHER2BinaryConfig)}
    cfg = PhenoHER2BinaryConfig(**{k: blob["config"][k] for k in names if k in blob["config"]})
    model = PhenoHER2Binary(cfg)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval().to(device)
    return model, cfg, int(blob.get("seed", 0))


def load_slide(features_dir: Path, slide_id: str, max_patches: int, seed: int):
    rng = np.random.default_rng(seed)
    with h5py.File(features_dir / f"{slide_id}.h5", "r") as f:
        x = f["features"][:]
    if len(x) > max_patches:
        idx = rng.choice(len(x), max_patches, replace=False)
        idx.sort()
        x = x[idx]
    return torch.from_numpy(x).float().unsqueeze(0)


@torch.no_grad()
def analyze(
    model,
    cfg,
    seed: int,
    features_dir: Path,
    diagnostics_json: Path,
    slide_id: str,
    max_patches: int,
    device: torch.device,
) -> dict:
    with diagnostics_json.open() as f:
        diag = json.load(f)
    corrs = np.array(diag["label_correlation_pearson_per_proto"], dtype=float)
    K = len(corrs)
    heroes = np.argsort(-np.abs(corrs))[: min(4, K)].tolist()

    P = F.normalize(model.prototypes.detach(), dim=-1)
    G = (P @ P.T).cpu().numpy()
    off = G[~np.eye(K, dtype=bool)]

    x = load_slide(features_dir, slide_id, max_patches, seed).to(device)
    out = model(x)
    sa = out["soft_assign"][0].cpu().numpy()
    pa = out["patch_attn"][0].cpu().numpy()
    if K > 1:
        C_attn = np.corrcoef(pa.T)
        C_route = np.corrcoef(sa.T)
        mean_abs_corr_attn = float(np.mean(np.abs(C_attn[~np.eye(K, dtype=bool)])))
    else:
        C_attn = np.array([[1.0]])
        C_route = np.array([[1.0]])
        mean_abs_corr_attn = 0.0
    top = max(1, pa.shape[0] // 10)
    jacc = {}
    hero_attn_corrs = {}
    for i, a in enumerate(heroes):
        for b in heroes[i + 1 :]:
            sa_set = set(np.argsort(-pa[:, a])[:top].tolist())
            sb_set = set(np.argsort(-pa[:, b])[:top].tolist())
            jacc[f"P{a}_P{b}"] = len(sa_set & sb_set) / max(1, len(sa_set | sb_set))
            hero_attn_corrs[f"P{a}_P{b}"] = float(C_attn[a, b]) if K > 1 else 1.0

    return {
        "K": K,
        "checkpoint": str(diag.get("checkpoint", "")),
        "prototype_cosine_offdiag_mean": float(off.mean()),
        "prototype_cosine_offdiag_max": float(off.max()),
        "label_correlation_heroes": {f"P{k}": float(corrs[k]) for k in heroes},
        "heroes": heroes,
        "slide_id": slide_id,
        "patch_attn_mean_abs_offdiag_corr": mean_abs_corr_attn,
        "hero_patch_attn_pairwise_corr": hero_attn_corrs,
        "hero_top10pct_jaccard": jacc,
        "routing_hero_mean_abs_corr": float(
            np.mean(np.abs(C_route[np.ix_(heroes, heroes)][~np.eye(len(heroes), dtype=bool)]))
        ),
        "usage_entropy_mean": float(diag.get("slide_usage_entropy_mean", float("nan"))),
    }


def plot_cosine_heatmap(model, out_path: Path, heroes: list[int]) -> None:
    P = F.normalize(model.prototypes.detach(), dim=-1)
    G = (P @ P.T).cpu().numpy()
    K = G.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(G, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Prototype cosine similarity (trained vectors)")
    ax.set_xlabel("Prototype")
    ax.set_ylabel("Prototype")
    plt.colorbar(im, ax=ax, fraction=0.046)
    for k in heroes:
        ax.axhline(k, color="lime", lw=0.5, alpha=0.5)
        ax.axvline(k, color="lime", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--diagnostics_json", type=Path, required=True)
    ap.add_argument("--features_dir", type=Path, required=True)
    ap.add_argument("--slide_id", default="201")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model, cfg, seed = load_model(args.checkpoint, device)
    report = analyze(
        model,
        cfg,
        seed,
        args.features_dir,
        args.diagnostics_json,
        args.slide_id,
        args.max_patches,
        device,
    )
    out_json = args.out_dir / "separation_analysis.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    plot_cosine_heatmap(model, args.out_dir / "prototype_cosine_heatmap.png", report["heroes"])
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

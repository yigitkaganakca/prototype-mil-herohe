"""Seaborn-styled prototype patch montage for report figures."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig
from herohe.gp2.scripts.herohe_wsi_paths import features_dir, resolve_wsi
from herohe.gp2.scripts.interp_viz_utils import active_heads, hard_assign, rank_patches_per_head


def pick_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(PhenoHER2BinaryConfig)}
    cfg = PhenoHER2BinaryConfig(**{k: blob["config"][k] for k in names if k in blob["config"]})
    model = PhenoHER2Binary(cfg).eval().to(device)
    model.load_state_dict(blob["model_state"], strict=True)
    seed = int(blob.get("seed", 0))
    return model, cfg, seed


def read_patch(wsi: Path, x: int, y: int, attrs: dict, size: int) -> np.ndarray:
    from openslide import OpenSlide

    ps0 = int(float(attrs.get("patch_size_level0", 256)))
    with OpenSlide(str(wsi)) as slide:
        im = slide.read_region((x, y), 0, (ps0, ps0)).convert("RGB")
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im)


@torch.no_grad()
def render_montage(
    checkpoint: Path,
    slide_id: str,
    split: str,
    out_path: Path,
    top_m: int = 6,
    tile_px: int = 128,
    max_patches: int = 4096,
    device_name: str = "mps",
    dpi: int = 200,
) -> dict:
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    device = pick_device(device_name)
    model, cfg, seed = load_model(checkpoint, device)
    K = cfg.num_prototypes

    fdir = features_dir(split)
    wsi = resolve_wsi(slide_id, split=split)
    with h5py.File(fdir / f"{slide_id}.h5") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)
        coords = np.asarray(f["coords"][:], dtype=np.int64)
        attrs = dict(f["coords"].attrs)

    idx = np.arange(len(feats))
    if len(feats) > max_patches:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(feats), max_patches, replace=False))
        feats, coords = feats[idx], coords[idx]

    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    out = model(x)
    sa = out["soft_assign"][0].cpu().numpy()
    pa = out["patch_attn"][0].cpu().numpy()
    N = pa.shape[0]
    hard = hard_assign(sa)
    hard_frac = np.bincount(hard, minlength=K) / len(hard)
    pta = out.get("phen_token_attn")
    phen_w = pta[0].cpu().numpy() if pta is not None else None
    phen_active = out.get("phenotype_active")
    if phen_active is not None:
        phen_active = phen_active[0].cpu().numpy().astype(bool)
    else:
        phen_active = np.ones(K, dtype=bool)

    heads = active_heads(phen_active, hard_frac, phen_w)
    if not heads:
        heads = list(range(K))
    n_rows = len(heads)
    top_m = min(top_m, N)

    palette = sns.color_palette("tab10", K)
    fig, axes = plt.subplots(n_rows, top_m, figsize=(top_m * 1.35, n_rows * 1.35), facecolor="white")
    if n_rows == 1:
        axes = np.array([axes])
    if top_m == 1:
        axes = axes.reshape(n_rows, 1)

    meta = {"slide_id": slide_id, "split": split, "wsi": str(wsi), "K": K, "top_m": top_m, "patches": []}

    for row, k in enumerate(heads):
        order = rank_patches_per_head(hard, pa, k, top_m)
        row_entry = {"prototype": k, "tiles": []}
        for j in range(top_m):
            ax = axes[row, j]
            if j >= len(order):
                ax.axis("off")
                continue
            pi = int(order[j])
            cx, cy = int(coords[pi, 0]), int(coords[pi, 1])
            try:
                tile = read_patch(wsi, cx, cy, attrs, tile_px)
            except Exception:
                tile = np.full((tile_px, tile_px, 3), 220, dtype=np.uint8)
                ax.text(0.5, 0.5, "read err", ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.imshow(tile)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(palette[k])
                spine.set_linewidth(2.5 if j == 0 else 1.2)
            if j == 0:
                hf_k = hard_frac[k] * 100
                w_k = phen_w[k] if phen_w is not None else float("nan")
                ax.set_ylabel(f"P{k}\n{hf_k:.0f}% · ω={w_k:.2f}", fontsize=9, fontweight="semibold", color=palette[k])
            if row == 0:
                ax.set_title(f"#{j+1}", fontsize=8, pad=4)
            row_entry["tiles"].append({
                "rank": j,
                "patch_idx": pi,
                "attn": float(pa[pi, k]),
                "x": cx,
                "y": cy,
            })
        meta["patches"].append(row_entry)

    subtitle = ""
    if phen_w is not None:
        subtitle = " · ".join(f"P{k}={phen_w[k]:.2f}" for k in heads if phen_w[k] > 0.01)
    dom = int(hard_frac.argmax())
    fig.suptitle(
        f"Hard-partition patch montage · slide {slide_id} ({split})\n"
        f"Ranked within cluster (hard==k) by patch attention · {subtitle}",
        fontsize=11, fontweight="semibold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta["output"] = str(out_path)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--slide_id", default="303")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top_m", type=int, default=6)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    meta = render_montage(
        args.checkpoint, args.slide_id, args.split, args.out,
        top_m=args.top_m, device_name=args.device,
    )
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

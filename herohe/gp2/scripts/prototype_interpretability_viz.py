"""Cohort prototype–label correlations + WSI patch-attention heatmaps (AP / PhenoBIN).

Example:

    python herohe/gp2/scripts/prototype_interpretability_viz.py \\
        --checkpoint herohe/gp2/runs/phenobin_binary_ap_fold0/fold_0/best.pt \\
        --features_dir herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2 \\
        --diagnostics_json herohe/gp2/runs/phenobin_binary_ap_fold0/prototype_diagnostics_all360.json \\
        --slides 201,422 \\
        --out_dir herohe/gp2/runs/phenobin_binary_ap_fold0/interpretability_viz
"""

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
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from PIL import Image

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig
from herohe.gp2.models.dataset import load_herohe_binary_labels

HERO_COLORS = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c", "#8c564b", "#e377c2", "#17becf"]


def heroes_from_diagnostics(diag: dict, n: int = 4) -> list[int]:
    corrs = np.array(diag["label_correlation_pearson_per_proto"], dtype=float)
    order = np.argsort(-np.abs(corrs))
    return order[: min(n, len(order))].tolist()


def hero_color(k: int, heroes: list[int]) -> str:
    if k in heroes:
        return HERO_COLORS[heroes.index(k) % len(HERO_COLORS)]
    return "#888888"


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
    model.eval()
    model.to(device)
    seed = int(blob.get("seed", 0))
    return model, cfg, seed


def load_slide_bag(features_dir: Path, slide_id: str, max_patches: int, seed: int):
    path = features_dir / f"{slide_id}.h5"
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        feats = f["features"][:]
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    if len(feats) > max_patches:
        idx = rng.choice(len(feats), max_patches, replace=False)
        idx.sort()
        feats = feats[idx]
        coords = coords[idx]
    return feats, coords, attrs


def plot_cohort_correlations(diag_json: Path, out_dir: Path) -> tuple[Path, list[int]]:
    with diag_json.open() as f:
        d = json.load(f)
    corrs = np.array(d["label_correlation_pearson_per_proto"], dtype=float)
    pos_minus_neg = np.array(d.get("usage_pos_minus_neg_per_proto", [0] * len(corrs)), dtype=float)
    K = len(corrs)
    heroes = heroes_from_diagnostics(d, n=min(4, K))
    x = np.arange(K)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

    colors = [hero_color(i, heroes) for i in range(K)]

    ax = axes[0]
    bars = ax.bar(x, corrs, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i}" for i in x], rotation=90, fontsize=7)
    ax.set_ylabel("Pearson r (proto usage vs HER2+ label)")
    ax.set_title(
        f"Cohort prototype–label signal (n={d.get('slides_used', '?')} slides, K={K})\n"
        f"Top-|r| heroes: {', '.join(f'P{k}' for k in heroes)} highlighted"
    )
    for k in heroes:
        ax.annotate(
            f"P{k}\nr={corrs[k]:+.2f}",
            xy=(k, corrs[k]),
            xytext=(k, corrs[k] + (0.06 if corrs[k] >= 0 else -0.08)),
            ha="center",
            fontsize=8,
            color=hero_color(k, heroes),
            fontweight="bold",
        )
    legend_patches = [Patch(facecolor=hero_color(k, heroes), label=f"P{k}") for k in heroes]
    legend_patches.append(Patch(facecolor="#888888", label="other protos"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    ax2 = axes[1]
    ax2.bar(x, pos_minus_neg * 1000, color=colors, edgecolor="black", linewidth=0.3)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlim(-0.5, K - 0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{i}" for i in x], rotation=90, fontsize=7)
    ax2.set_ylabel("mean usage diff (pos−neg) ×1000")
    ax2.set_xlabel("Prototype index")
    ax2.set_title("HER2+ vs HER2− mean patch-assignment gap (same heroes highlighted)")

    fig.tight_layout()
    out = out_dir / "cohort_proto_label_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, heroes


def find_thumbnail(slide_id: str, trident_root: Path) -> Path | None:
    for rel in (
        f"20x_256px_0px_overlap/visualization/{slide_id}.jpg",
        f"thumbnails/{slide_id}.jpg",
    ):
        p = trident_root / rel
        if p.is_file():
            return p
    return None


def rasterize_attention(
    coords: np.ndarray,
    values: np.ndarray,
    attrs: dict,
    thumb_size: tuple[int, int],
) -> np.ndarray:
    """Paint patch attention onto a thumbnail-sized accumulation grid."""
    w0 = float(attrs["level0_width"])
    h0 = float(attrs["level0_height"])
    patch = float(attrs["patch_size_level0"])
    tw, th = thumb_size
    sx, sy = tw / w0, th / h0
    heat = np.zeros((th, tw), dtype=np.float64)
    count = np.zeros((th, tw), dtype=np.float64)

    pw = max(1, int(round(patch * sx)))
    ph = max(1, int(round(patch * sy)))

    for (cx, cy), v in zip(coords, values):
        x0 = int(cx * sx)
        y0 = int(cy * sy)
        x1 = min(tw, x0 + pw)
        y1 = min(th, y0 + ph)
        x0 = max(0, x0)
        y0 = max(0, y0)
        if x1 <= x0 or y1 <= y0:
            continue
        heat[y0:y1, x0:x1] += float(v)
        count[y0:y1, x0:x1] += 1.0

    mask = count > 0
    heat[mask] /= count[mask]
    return heat


def overlay_heatmap(
    thumb_path: Path,
    heat: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str = "jet",
    alpha: float = 0.55,
) -> None:
    thumb = np.array(Image.open(thumb_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(thumb)
    axes[0].set_title("WSI thumbnail (TRIDENT)")
    axes[0].axis("off")

    vmax = np.percentile(heat[heat > 0], 99) if np.any(heat > 0) else 1.0
    norm = Normalize(vmin=0, vmax=max(vmax, 1e-8))
    axes[1].imshow(thumb)
    im = axes[1].imshow(heat, cmap=cmap, alpha=alpha, norm=norm)
    axes[1].set_title("Patch attention overlay")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="attention weight")

    axes[2].imshow(heat, cmap=cmap, norm=norm)
    axes[2].set_title("Attention heatmap only")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slide_heatmaps(
    model,
    cfg,
    seed: int,
    features_dir: Path,
    trident_root: Path,
    slide_id: str,
    label: int,
    out_dir: Path,
    device: torch.device,
    max_patches: int,
    heroes: list[int],
) -> list[Path]:
    feats, coords, attrs = load_slide_bag(features_dir, slide_id, max_patches, seed)
    x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = model.predict(x)
    pa = out["patch_attn"][0].float().cpu().numpy()
    prob = float(pred["prob_positive"])

    thumb_path = find_thumbnail(slide_id, trident_root)
    if thumb_path is None:
        raise FileNotFoundError(f"No TRIDENT thumbnail for slide {slide_id}")

    thumb = Image.open(thumb_path)
    thumb_size = thumb.size
    outs: list[Path] = []

    total_attn = pa.sum(axis=1)
    total_attn = total_attn / (total_attn.max() + 1e-8)
    heat_total = rasterize_attention(coords, total_attn, attrs, thumb_size)
    tag = "HER2+" if label == 1 else "HER2-"
    p_total = out_dir / f"heatmap_slide{slide_id}_total_patch_attn.png"
    overlay_heatmap(
        thumb_path,
        heat_total,
        f"Slide {slide_id} ({tag})  P(HER2+)={prob:.3f}  — sum of patch_attn over all {cfg.num_prototypes} protos",
        p_total,
    )
    outs.append(p_total)

    n_heroes = min(4, len(heroes))
    heroes = heroes[:n_heroes]
    nrows = 2 if n_heroes > 2 else 1
    ncols = 2 if n_heroes > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes_flat = np.array(axes).reshape(-1)
    for ax_i, k in enumerate(heroes):
        v = pa[:, k]
        v = v / (v.max() + 1e-8)
        heat_k = rasterize_attention(coords, v, attrs, thumb_size)
        thumb_rgb = np.array(thumb.convert("RGB"))
        ax = axes_flat[ax_i]
        ax.imshow(thumb_rgb)
        vmax = np.percentile(heat_k[heat_k > 0], 99) if np.any(heat_k > 0) else 1.0
        ax.imshow(heat_k, cmap="jet", alpha=0.55, vmin=0, vmax=max(vmax, 1e-8))
        ax.set_title(f"P{k}  max attn={pa[:, k].max():.2e}")
        ax.axis("off")
    for ax in axes_flat[n_heroes:]:
        ax.axis("off")
    fig.suptitle(
        f"Slide {slide_id} ({tag})  P(HER2+)={prob:.3f}  — top-|r| hero patch_attn",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    p_heroes = out_dir / f"heatmap_slide{slide_id}_hero_protos.png"
    fig.savefig(p_heroes, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outs.append(p_heroes)
    return outs


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--features_dir", type=Path, required=True)
    ap.add_argument("--diagnostics_json", type=Path, required=True)
    ap.add_argument("--labels_csv", type=Path, default=_REPO / "herohe/Training (ground truth).csv")
    ap.add_argument("--trident_root", type=Path, default=_REPO / "herohe/gp2/results_trident_mac_full")
    ap.add_argument("--slides", default="201,422")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_patches", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, cfg, seed = load_model(args.checkpoint, device)
    labels_df = load_herohe_binary_labels(str(args.labels_csv))
    label_map = dict(zip(labels_df.slide_id.astype(str), labels_df.label.astype(int)))

    p1, heroes = plot_cohort_correlations(args.diagnostics_json, args.out_dir)
    print(f"Wrote {p1}  heroes={heroes}")

    for sid in args.slides.split(","):
        sid = sid.strip()
        if sid not in label_map:
            print(f"Skip {sid}: no label")
            continue
        paths = plot_slide_heatmaps(
            model,
            cfg,
            seed,
            args.features_dir,
            args.trident_root,
            sid,
            int(label_map[sid]),
            args.out_dir,
            device,
            args.max_patches,
            heroes,
        )
        for p in paths:
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()

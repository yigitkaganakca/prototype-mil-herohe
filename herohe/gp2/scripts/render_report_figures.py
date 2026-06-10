"""Generate all report figures for herohe/report/ (§4 + §5).

Usage:
    python herohe/gp2/scripts/render_report_figures.py all
    python herohe/gp2/scripts/render_report_figures.py results
    python herohe/gp2/scripts/render_report_figures.py interpretability
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from sklearn.decomposition import PCA

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.scripts.herohe_wsi_paths import resolve_wsi, trident_root as herohe_trident_root

FIG = _REPO / "herohe/report/figures"
FEAT = _REPO / "herohe/gp2/results_trident_mac_full/20x_256px_0px_overlap/features_virchow2"
FEAT_TEST = _REPO / "herohe/gp2/results_trident_test/20x_256px_0px_overlap/features_virchow2"
TRIDENT = _REPO / "herohe/gp2/results_trident_mac_full"
TRIDENT_TEST = _REPO / "herohe/gp2/results_trident_test"
DEFAULT_SLIDE = "304"
INTERP_RANKINGS = _REPO / "herohe/gp2/data/interp_slide_rankings.json"
LABELS = _REPO / "herohe/Training (ground truth).csv"
KHEAD_CKPT = _REPO / "herohe/gp2/runs/khead_token_abmil_hard_partition_ent0/fold_0/best.pt"
KHEAD_METRICS_BIN = _REPO / "herohe/gp2/runs/khead_hard_partition_medoid_proto_control/test_eval/metrics_medoid_proto_5fold.json"
KHEAD_METRICS_3C = _REPO / "herohe/gp2/runs/medoid_benchmark/tri_hard_token_L8/test_eval/metrics_tri_hard_token_L8_5fold.json"
ABMIL_CKPT = _REPO / "herohe/gp2/runs/abmil_phiher2fold_valloss/fold_0/best.pt"
AP_PROTO = _REPO / "herohe/gp2/data/prototypes_ap_phiher2fold_fold0_train_L8.pt"
PY = sys.executable


def recommended_slide(fallback: str = DEFAULT_SLIDE) -> str:
    if INTERP_RANKINGS.is_file():
        data = json.loads(INTERP_RANKINGS.read_text())
        return str(data.get("recommended_slide_id", fallback))
    return fallback

PROTO_COLORS = [
    "#d62728", "#ff7f0e", "#9467bd", "#1f77b4",
    "#2ca02c", "#8c564b", "#e377c2", "#17becf",
]


def _save(fig, path: Path, dpi: int = 160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def render_baselines_mechanism(out: Path):
    subprocess.run([
        PY, str(_REPO / "herohe/gp2/scripts/render_arch_baselines_figure.py"),
        "--out", str(out / "arch_baselines_mechanism.png"),
        "--dpi", "180",
    ], check=True)


def render_prototype_discovery(out: Path, slide_id: str = DEFAULT_SLIDE, seed: int = 42,
                               centers_path: Path | None = None,
                               out_name: str = "arch_prototypes_ap.png",
                               layout: str = "abc"):
    import seaborn as sns
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    from herohe.gp2.scripts.interp_viz_utils import pick_prototype_exemplar

    blob = torch.load(centers_path or AP_PROTO, map_location="cpu", weights_only=False)
    centers = F.normalize(blob["centers"].float(), dim=1).numpy()
    L = centers.shape[0]

    rng = np.random.default_rng(seed)
    feats_list, assign_list = [], []
    train_ids = blob.get("train_slide_ids", [])
    if isinstance(train_ids, torch.Tensor):
        train_ids = train_ids.tolist()
    ids = [str(x) for x in train_ids][:40] if train_ids else []
    if not ids:
        ids = sorted(p.stem for p in FEAT.glob("*.h5"))[:40]
    per_slide = 800
    for sid in ids:
        h5 = FEAT / f"{sid}.h5"
        if not h5.is_file():
            continue
        with h5py.File(h5, "r") as f:
            x = f["features"][:]
        if len(x) > per_slide:
            idx = rng.choice(len(x), per_slide, replace=False)
            x = x[idx]
        x_n = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        c_n = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        a = (x_n @ c_n.T).argmax(axis=1)
        feats_list.append(x)
        assign_list.append(a)
    X = np.vstack(feats_list)
    A = np.concatenate(assign_list)
    if len(X) > 12000:
        idx = rng.choice(len(X), 12000, replace=False)
        X, A = X[idx], A[idx]

    # Non-linear 2-D embedding (t-SNE) resolves transformer-feature clusters that a linear
    # PCA projection cannot; PCA is first used only to denoise to 50 components for speed.
    from sklearn.manifold import TSNE
    X50 = PCA(n_components=50, random_state=seed).fit_transform(X)
    xy = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca",
              learning_rate="auto").fit_transform(X50)

    sim = np.clip(centers @ centers.T, -1, 1)
    include_exemplars = layout != "ab"

    if include_exemplars:
        wsi = resolve_wsi(slide_id, split="test" if (FEAT_TEST / f"{slide_id}.h5").is_file() else "train")
        feat_h5 = FEAT_TEST / f"{slide_id}.h5" if (FEAT_TEST / f"{slide_id}.h5").is_file() else FEAT / f"{slide_id}.h5"
        with h5py.File(feat_h5, "r") as f:
            coords = f["coords"][:]
            attrs = dict(f["coords"].attrs)
            x_slide = f["features"][:]
        tile_px = 140
        exemplars: list[np.ndarray] = []
        for k in range(L):
            tile = pick_prototype_exemplar(coords, x_slide, centers, k, wsi, attrs, out_px=tile_px)
            if tile is None:
                tile = np.full((tile_px, tile_px, 3), 240, dtype=np.uint8)
            exemplars.append(tile)
        mosaic = np.zeros((tile_px, tile_px * L, 3), dtype=np.uint8)
        for k in range(L):
            mosaic[:, k * tile_px : (k + 1) * tile_px] = exemplars[k]

    if layout == "ab":
        fig = plt.figure(figsize=(12.5, 4.8), facecolor="white")
        gs = GridSpec(1, 2, wspace=0.16)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
    else:
        fig = plt.figure(figsize=(15, 9.5), facecolor="white")
        gs = GridSpec(2, 2, height_ratios=[1.25, 0.72], hspace=0.30, wspace=0.14)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])

    for k in range(L):
        m = A == k
        ax0.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.4, c=PROTO_COLORS[k % len(PROTO_COLORS)], label=f"P{k}")
    ax0.set_title("(a) Training-fold patch t-SNE (Virchow2), colored by prototype assignment",
                  fontweight="bold", fontsize=11, loc="center")
    ax0.set_xlabel("t-SNE 1")
    ax0.set_ylabel("t-SNE 2")
    ax0.legend(markerscale=3, fontsize=7, ncol=2, loc="upper right", frameon=True)

    sns.heatmap(sim, ax=ax1, cmap="vlag", center=0, vmin=-1, vmax=1, square=True,
                xticklabels=[f"P{i}" for i in range(L)], yticklabels=[f"P{i}" for i in range(L)],
                cbar_kws={"shrink": 0.82, "label": "cosine sim."})
    ax1.set_title("(b) Prototype cosine similarity ($L=8$)", fontweight="bold", fontsize=11, loc="center")

    if include_exemplars:
        ax2 = fig.add_subplot(gs[1, :])
        ax2.imshow(mosaic, extent=[0, L, 1, 0], aspect="equal")
        ax2.set_xlim(0, L)
        ax2.set_ylim(1, 0)
        ax2.set_xticks(np.arange(L) + 0.5)
        ax2.set_xticklabels([f"P{k}" for k in range(L)], fontsize=10, fontweight="semibold")
        ax2.set_yticks([])
        for k in range(L):
            ax2.add_patch(Rectangle(
                (k, 0), 1, 1, fill=False, edgecolor=PROTO_COLORS[k], lw=2.5,
            ))
        ax2.set_title(
            f"(c) Nearest-prototype exemplar per head (slide {slide_id}, medoid patch)",
            fontweight="bold", fontsize=11, pad=10, loc="center",
        )
        for spine in ax2.spines.values():
            spine.set_visible(False)
        fig.suptitle("Fold-wise prototype discovery: AP$\\,\\to\\,$$k$-means$\\,\\to\\,$real-patch medoid ($L=8$)",
                     fontsize=13, fontweight="bold", y=0.98)
        fig.subplots_adjust(top=0.91, bottom=0.08)
    else:
        fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.12)

    _save(fig, out / out_name, dpi=200)


def _load_json_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "results" in data:
        return data["results"][0]
    return data


def _auc_from_metrics(path: Path, fallback: float | None = None) -> float | None:
    if not path.is_file():
        return fallback
    m = _load_json_metrics(path)
    return float(m.get("AUC", m.get("auc_positive", fallback)))


def _encoder_ablation_rows() -> list[tuple[str, float, float | None]]:
    """Return (label, virchow2_auc, resnet50_auc) for encoder ablation panel."""
    v2 = _REPO / "herohe/gp2/runs/test_eval_mil/binary_5fold_s42"
    r50 = _REPO / "herohe/gp2/runs/test_eval_mil/resnet50_binary_5fold_s42"
    khead_r50 = _REPO / "herohe/gp2/runs/khead_token_abmil_hard_partition_ent0_resnet50/test_eval"
    return [
        ("PhenoBIN", _auc_from_metrics(KHEAD_METRICS_BIN, 0.826),
         _auc_from_metrics(khead_r50 / "metrics_khead_token_abmil_hard_partition_ent0_resnet50_5fold.json",
                           _auc_from_metrics(_REPO / "herohe/gp2/runs/test_eval_phenonly/khead_resnet50_tuned/metrics_khead_resnet50_tuned_5fold.json", 0.547))),
        ("CLAM", _auc_from_metrics(v2 / "summary_clam_binary_5fold_s42_valloss_5fold.json", 0.769),
         _auc_from_metrics(r50 / "summary_clam_resnet50_5fold_s42_valloss_5fold.json", 0.562)),
        ("ABMIL", _auc_from_metrics(v2 / "summary_abmil_binary_5fold_s42_valloss_5fold.json", 0.762),
         _auc_from_metrics(r50 / "summary_abmil_resnet50_5fold_s42_valloss_5fold.json", 0.552)),
        ("TransMIL", _auc_from_metrics(v2 / "summary_transmil_binary_5fold_s42_valloss_5fold.json", 0.765),
         _auc_from_metrics(r50 / "summary_transmil_resnet50_5fold_s42_valloss_5fold.json", 0.487)),
    ]


def _bar_labels(ax, bars, fmt="{:.3f}", fs=7, dy=0.012):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs)


def render_results_figures(out: Path):
    base = _REPO / "herohe/gp2/runs/test_eval_mil"
    # Binary bar chart removed from report (tables sufficient).

    # Medoid k-ablation (150-slide ensemble); see runs/uncertainty/uncertainty.json
    k_data = [
        {"K": 4, "test_auc": 0.846, "test_macro_f1": 0.763},
        {"K": 8, "test_auc": 0.826, "test_macro_f1": 0.732},
        {"K": 16, "test_auc": 0.766, "test_macro_f1": 0.679},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white")

    ax = axes[0]
    k_data_sorted = sorted(k_data, key=lambda r: r["K"])
    ks = [str(r["K"]) for r in k_data_sorted]
    auc_k = [r["test_auc"] for r in k_data_sorted]
    f1_k = [r["test_macro_f1"] for r in k_data_sorted]
    bars = ax.bar(ks, auc_k, color="#5B9BD5", edgecolor="white", label="Test AUC")
    _bar_labels(ax, bars, dy=0.015)
    ax2 = ax.twinx()
    ax2.plot(ks, f1_k, "o-", color="#C0504D", linewidth=2, markersize=7, label="Macro-F1")
    for xi, yi in zip(ks, f1_k):
        ax2.text(xi, yi + 0.008, f"{yi:.3f}", ha="center", fontsize=7, color="#C0504D")
    ax.set_title("$L$ ablation", fontweight="bold")
    ax.set_xlabel("Prototype count $L$")
    ax.set_ylabel("Test AUC")
    ax2.set_ylabel("Macro-F1", color="#C0504D")
    ax2.set_ylim(0.66, 0.79)
    ax.set_ylim(0.74, 0.87)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ax = axes[1]
    pool_data = [
        {"pool": "hard part.\n+ token ABMIL", "test_auc": 0.826},
        {"pool": "indep.\n+ token ABMIL", "test_auc": 0.829},
        {"pool": "mean", "test_auc": 0.768},
    ]
    pools = [r["pool"] for r in pool_data]
    auc_p = [r["test_auc"] for r in pool_data]
    bars = ax.bar(pools, auc_p, color=["#2E5090", "#4F8F5F", "#E8A317"], edgecolor="white", width=0.55)
    _bar_labels(ax, bars, dy=0.015)
    ax.set_title("Readout routing", fontweight="bold")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.74, 0.86)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ax = axes[2]
    enc_rows = _encoder_ablation_rows()
    labels_e = [r[0] for r in enc_rows]
    v2_auc = [r[1] for r in enc_rows]
    r50_auc = [r[2] if r[2] is not None else 0.0 for r in enc_rows]
    xv = np.arange(len(labels_e))
    b1 = ax.bar(xv - 0.18, v2_auc, 0.36, label="Virchow2", color="#2E5090", edgecolor="white")
    b2 = ax.bar(xv + 0.18, r50_auc, 0.36, label="ResNet-50", color="#A5A5A5", edgecolor="white")
    _bar_labels(ax, b1, dy=0.02)
    _bar_labels(ax, b2, dy=0.02)
    for xi, (_, _, r50) in enumerate(enc_rows):
        if r50 is None:
            ax.text(xi + 0.18, 0.02, "pending", ha="center", fontsize=6, color="#666666", rotation=90)
    ax.set_xticks(xv)
    ax.set_xticklabels(labels_e)
    ax.set_title("Encoder ablation (AUC)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("khead ablation summary", fontsize=12, fontweight="bold")
    fig.tight_layout()
    # NOTE: ablation summary figure removed from the report (ablations are reported as
    # Tables tab:k-ablation / tab:routing-ablation / tab:encoder-ablation). Kept here for
    # exploratory use only; not written to the figures dir.
    plt.close(fig)

    confusion_specs = [
        ("khead", "PhenoBIN khead", KHEAD_METRICS_3C),
        ("abmil", "ABMIL", base / "valieris3_5fold_s42/summary_abmil_valieris3_5fold_s42_valloss_5fold.json"),
        ("clam", "CLAM", base / "valieris3_5fold_s42/summary_clam_valieris3_5fold_s42_valloss_5fold.json"),
        ("transmil", "TransMIL", base / "valieris3_5fold_s42/summary_transmil_valieris3_5fold_s42_valloss_5fold.json"),
    ]
    panels = []
    for tag, label, summ_path in confusion_specs:
        if not summ_path.is_file():
            continue
        data = _load_json_metrics(summ_path)
        panels.append((tag, label, np.array(data["confusion_matrix"])))

    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), facecolor="white")
        axes = axes.flatten()
        for ax, (tag, label, cm) in zip(axes, panels):
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(["Neg", "Low", "High"])
            ax.set_yticklabels(["Neg", "Low", "High"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] >= cm.max() * 0.6 else "black", fontsize=9)
            ax.set_title(label, fontweight="bold", fontsize=10)
        for ax in axes[len(panels):]:
            ax.axis("off")
        fig.suptitle("Three-class test confusion matrices ($n=149$)", fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, out / "results_confusion_3class_grid.png")

    for tag, label, summ_path in confusion_specs:
        if not summ_path.is_file():
            continue
        data = _load_json_metrics(summ_path)
        cm = np.array(data["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.5, 4), facecolor="white")
        im = ax.imshow(cm, cmap="Blues")
        classes = ["Neg", "Low", "High"]
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() * 0.6 else "white")
        ax.set_title(f"3-class confusion ({tag})", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
        _save(fig, out / f"results_confusion_3class_{tag}.png")


def run_interpretability(out: Path, device: str = "mps", slide_id: str = DEFAULT_SLIDE, split: str = "test"):
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    diag_json = meta / "prototype_diagnostics_ent0_fold0.json"

    subprocess.run([
        PY, str(_REPO / "herohe/gp2/scripts/prototype_diagnostics.py"),
        "--checkpoint", str(KHEAD_CKPT),
        "--features_dir", str(FEAT),
        "--labels_csv", str(LABELS),
        "--label_mode", "gt_binary",
        "--max_slides", "360",
        "--device", device,
    ], check=True, stdout=open(diag_json, "w"))

    interp_dir = meta / "interp_tmp"
    interp_dir.mkdir(exist_ok=True)
    feat_dir = FEAT_TEST if split == "test" else FEAT
    trident_root_path = herohe_trident_root(split)
    wsi_path = resolve_wsi(slide_id, split=split)

    subprocess.run([
        PY, str(_REPO / "herohe/gp2/scripts/prototype_interpretability_viz.py"),
        "--checkpoint", str(KHEAD_CKPT),
        "--features_dir", str(feat_dir),
        "--diagnostics_json", str(diag_json),
        "--labels_csv", str(LABELS),
        "--trident_root", str(trident_root_path),
        "--slides", slide_id,
        "--out_dir", str(interp_dir),
        "--device", device,
    ], check=True)

    for name in ("cohort_proto_label_correlation.png",):
        src = interp_dir / name
        if src.is_file():
            dst = out / f"interp_{name.replace('cohort_proto_label_correlation', 'cohort_proto_correlation')}"
            dst.write_bytes(src.read_bytes())
            print(f"Wrote {dst}")

    for pat in (f"heatmap_slide{slide_id}_*.png",):
        for src in interp_dir.glob(pat):
            dst = out / f"interp_{src.name}"
            dst.write_bytes(src.read_bytes())
            print(f"Wrote {dst}")

    subprocess.run([
        PY, str(_REPO / "herohe/gp2/scripts/render_interp_slide_figure.py"),
        "--checkpoint", str(KHEAD_CKPT),
        "--slide_id", slide_id,
        "--split", split,
        "--out", str(out / f"interp_slide{slide_id}_composite.png"),
        "--device", device,
    ], check=True)

    subprocess.run([
        PY, str(_REPO / "herohe/gp2/scripts/render_interp_montage.py"),
        "--checkpoint", str(KHEAD_CKPT),
        "--slide_id", slide_id,
        "--split", split,
        "--out", str(out / f"interp_montage_slide{slide_id}.png"),
        "--top_m", "4",
        "--device", device,
    ], check=True)

    subprocess.run([
        PY, str(_REPO / "herohe/gp2/scripts/prototype_separation_analysis.py"),
        "--checkpoint", str(KHEAD_CKPT),
        "--diagnostics_json", str(diag_json),
        "--features_dir", str(feat_dir),
        "--slide_id", slide_id,
        "--out_dir", str(meta / "separation"),
        "--device", device,
    ], check=True)
    src = meta / "separation/prototype_cosine_heatmap.png"
    if src.is_file():
        dst = out / "interp_prototype_cosine.png"
        dst.write_bytes(src.read_bytes())
        print(f"Wrote {dst}")



def render_attention_compare(out: Path, slide_id: str, device: str,
                             feat_dir: Path | None = None, trident_root: Path | None = None,
                             split: str = "test"):
    import seaborn as sns
    from herohe.gp2.models import PhenoHER2Binary, PhenoHER2BinaryConfig
    from herohe.gp2.scripts.eval_mil_baseline_test import load_mil_checkpoint

    sns.set_theme(style="white", context="paper", font_scale=1.05)

    def heatmap_from_attn(coords, attn, attrs, thumb_shape):
        w0, h0 = float(attrs["level0_width"]), float(attrs["level0_height"])
        patch = float(attrs["patch_size_level0"])
        th, tw = thumb_shape[1], thumb_shape[0]
        sx, sy = tw / w0, th / h0
        heat = np.zeros((th, tw))
        pw = max(1, int(round(patch * sx)))
        ph = max(1, int(round(patch * sy)))
        for (cx, cy), v in zip(coords, attn):
            x0, y0 = int(cx * sx), int(cy * sy)
            heat[y0:min(th, y0 + ph), x0:min(tw, x0 + pw)] = np.maximum(
                heat[y0:min(th, y0 + ph), x0:min(tw, x0 + pw)], float(v))
        if heat.max() > 0:
            heat /= heat.max()
        return heat

    feat_dir = feat_dir or FEAT
    trident_root = trident_root or TRIDENT
    thumb_path = trident_root / "thumbnails" / f"{slide_id}.jpg"
    thumb = np.array(Image.open(thumb_path).convert("RGB"))

    with h5py.File(feat_dir / f"{slide_id}.h5", "r") as f:
        feats = f["features"][:]
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)

    dev = torch.device("mps" if device == "mps" and torch.backends.mps.is_available() else "cpu")

    blob = torch.load(KHEAD_CKPT, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(PhenoHER2BinaryConfig)}
    cfg = PhenoHER2BinaryConfig(**{k: blob["config"][k] for k in names if k in blob["config"]})
    kmodel = PhenoHER2Binary(cfg).eval().to(dev)
    kmodel.load_state_dict(blob["model_state"], strict=True)
    x = torch.from_numpy(feats[:4096]).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        ko = kmodel(x)
    pa = ko["patch_attn"].squeeze(0).cpu().numpy()
    if pa.shape[0] != x.shape[1]:
        pa = pa.T
    k_attn = pa.sum(axis=0)
    k_attn = k_attn / (k_attn.max() + 1e-8)
    kheat = heatmap_from_attn(coords[:4096], k_attn, attrs, thumb.shape[:2][::-1])

    amodel, _, _, _ = load_mil_checkpoint(ABMIL_CKPT, dev)
    with torch.no_grad():
        ao = amodel(x)
    a_attn = ao["attn"].squeeze(0).cpu().numpy()
    a_attn = a_attn / (a_attn.max() + 1e-8)
    aheat = heatmap_from_attn(coords[:4096], a_attn, attrs, thumb.shape[:2][::-1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor="white")
    for ax, heat, title in zip(
        axes,
        [kheat, aheat],
        ["khead (Σ patch-attn over $L$ heads)", "ABMIL (global attention)"],
    ):
        ax.imshow(thumb)
        ax.imshow(heat, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        ax.set_title(f"{title}\nSlide {slide_id} ({split})", fontweight="semibold", fontsize=10)
        ax.axis("off")
    fig.suptitle("Attention overlay on TRIDENT thumbnail", fontsize=12, fontweight="bold")
    _save(fig, out / f"interp_attention_compare_slide{slide_id}.png", dpi=180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "architecture", "results", "interpretability"])
    ap.add_argument("--slide_id", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    subprocess.run([PY, str(_REPO / "herohe/gp2/scripts/rank_interp_slides.py")], check=True)
    slide_id = args.slide_id or recommended_slide()

    trident_job = TRIDENT_TEST if args.split == "test" else TRIDENT

    if args.cmd in ("all", "architecture"):
        render_baselines_mechanism(FIG)
        render_prototype_discovery(FIG, slide_id=slide_id)
        subprocess.run([
            "bash", str(_REPO / "herohe/gp2/scripts/run_architecture_figures.sh"),
        ], check=True, env={**dict(__import__("os").environ),
                              "SLIDE_ID": slide_id, "SPLIT": args.split,
                              "CKPT": str(KHEAD_CKPT),
                              "TRIDENT_JOB": str(trident_job)})
        subprocess.run([
            PY, str(_REPO / "herohe/gp2/scripts/render_architecture_pipeline_figure.py"),
            "--slide_id", slide_id,
            "--split", args.split,
            "--trident_job_dir", str(trident_job),
            "--out_dir", str(FIG),
            "--dpi", "220",
        ], check=True)

    if args.cmd in ("all", "results"):
        render_results_figures(FIG)

    if args.cmd in ("all", "interpretability"):
        run_interpretability(FIG, device=args.device, slide_id=slide_id, split=args.split)


if __name__ == "__main__":
    main()

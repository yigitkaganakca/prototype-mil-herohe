"""Build hierarchical prototypes (per-slide AP + global condensation) from training folds.

Stage 1 (always AP): AP on each slide's subsampled patches → slide exemplars (real
patch medoids). Follows PhiHER2 ``utils/cluster_utils.py`` + ``cfgs/HEROHE.yaml``: PMIL
similarity ``exp(-λ·d/max(d))``, ``preference=0``, ``damping=0.5``, up to 5000
patches/slide.

Stage 2 (``--stage2_method``): condense pooled stage-1 exemplars into the global
prototype set — either ``kmeans`` (exactly K = ``--target_L`` centers) or ``ap`` (a
second AP pass, data-driven L).

REPORTED RESULTS: the run scripts always pass ``--stage2_method kmeans --target_L L``
(L = 8 primary; L ∈ {4, 8, 16} for the prototype-count ablation). So the prototypes
used for the reported tables are Stage-1 AP → Stage-2 MiniBatchKMeans(K = L). The
``ap`` stage-2 path is the PhiHER2-faithful variant, retained for reference but not
used for the reported numbers. (The CLI default is ``ap``; the run scripts override it.)

Run on **training slides only** (single CSV or one CV fold's train split). Output .pt
is loaded by ``train_phenobin_mil.py --prototypes``; prototypes are frozen by default
(PhiHER2 Cluster-PT).

Examples
--------
Reported recipe — fold 0 train pool, k-means stage-2 to exactly L=8 (no val leakage):

    python herohe/gp2/scripts/init_prototypes_ap.py \\
        --features_dir .../features_virchow2 \\
        --folds_csv herohe/gp2/data/folds_phiher2_binary_s42.csv \\
        --val_fold 0 --stage2_method kmeans --target_L 8 \\
        --output herohe/gp2/data/prototypes_ap_phiher2fold_fold0_train_L8.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[3]
sys.path.insert(0, str(_REPO))

from herohe.gp2.prototype_discovery import (
    collect_patches_per_slide,
    hierarchical_ap_prototypes,
    save_hierarchical_ap_checkpoint,
    slide_ids_from_csv,
    stage2_ap_from_pool,
    stage2_kmeans_from_pool,
    train_slide_ids_from_folds,
)
from herohe.gp2.prototype_discovery.ap_hierarchical import HierarchicalAPResult


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_dir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--slide_ids_csv", help="CSV with slide_id or wsi column (training slides only)")
    src.add_argument(
        "--folds_csv",
        help="Use all folds except --val_fold as the training pool (recommended for CV)",
    )
    ap.add_argument(
        "--val_fold",
        type=int,
        default=None,
        help="Required with --folds_csv: held-out fold index (prototypes fit on other folds)",
    )
    ap.add_argument(
        "--patches_per_slide",
        type=int,
        default=5000,
        help="Max patches sampled per slide before stage-1 AP (PhiHER2: 5000)",
    )
    ap.add_argument(
        "--preference",
        type=float,
        default=0.0,
        help="AP preference for stage 1 (PhiHER2 HEROHE.yaml: 0)",
    )
    ap.add_argument(
        "--stage2_preference",
        type=float,
        default=None,
        help="Override AP preference for stage 2 only (default: same as --preference)",
    )
    ap.add_argument(
        "--target_L",
        type=int,
        default=None,
        help="Target global L: for ap=approximate via preference search; for kmeans=exact K",
    )
    ap.add_argument(
        "--stage2_method",
        choices=["ap", "kmeans"],
        default="ap",
        help="Stage-2 condensation on pooled exemplars (default: ap). "
        "Use kmeans with --target_L for exact L in {4,8,...}.",
    )
    ap.add_argument(
        "--cache_stage2_pool",
        default=None,
        help="Save pooled stage-1 exemplars (.npy) for reuse",
    )
    ap.add_argument(
        "--reuse_stage2_pool",
        default=None,
        help="Skip stage-1; load pooled exemplars and run stage-2 only",
    )
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument(
        "--lamb",
        type=float,
        default=0.25,
        help="PMIL similarity scale exp(-λ·d/max(d)) (PhiHER2 HEROHE.yaml: 0.25)",
    )
    ap.add_argument("--min_patches_per_slide", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help="Output .pt path")
    args = ap.parse_args()

    if args.folds_csv is not None:
        if args.val_fold is None:
            ap.error("--val_fold is required when using --folds_csv")
        slide_ids = train_slide_ids_from_folds(args.folds_csv, args.val_fold)
        split_note = f"folds_csv val_fold={args.val_fold} train_n={len(slide_ids)}"
    else:
        slide_ids = slide_ids_from_csv(args.slide_ids_csv)
        split_note = f"slide_ids_csv n={len(slide_ids)}"

    print(
        f"[init_prototypes_ap] {split_note}; patches_per_slide={args.patches_per_slide}  "
        f"preference={args.preference}  damping={args.damping}  lamb={args.lamb}"
    )
    if args.reuse_stage2_pool:
        slide_patches = {}
        missing = []
    else:
        slide_patches, missing = collect_patches_per_slide(
            args.features_dir,
            slide_ids,
            patches_per_slide=args.patches_per_slide,
            seed=args.seed,
        )
        if missing:
            print(f"[init_prototypes_ap] WARNING: {len(missing)} slides missing .h5; e.g. {missing[:5]}")
    if not slide_patches and not args.reuse_stage2_pool:
        raise SystemExit("No patch data collected.")

    if args.reuse_stage2_pool:
        if not slide_ids:
            ap.error("--reuse_stage2_pool requires --folds_csv or --slide_ids_csv for metadata")
        pool_path = Path(args.reuse_stage2_pool)
        stage2_in = np.load(pool_path)
        print(f"[init_prototypes_ap] loaded stage2 pool {pool_path} shape={stage2_in.shape}")
        if args.stage2_method == "kmeans":
            if args.target_L is None:
                ap.error("--stage2_method kmeans requires --target_L")
            centers = stage2_kmeans_from_pool(stage2_in, int(args.target_L), seed=args.seed)
            s2_pref = float("nan")
        else:
            s2_pref_arg = args.stage2_preference if args.stage2_preference is not None else args.preference
            centers, s2_pref = stage2_ap_from_pool(
                stage2_in,
                preference=None if args.target_L is not None else s2_pref_arg,
                target_L=args.target_L,
                damping=args.damping,
                lamb=args.lamb,
                seed=args.seed,
            )
        result = HierarchicalAPResult(
            centers=centers,
            stage1=[],
            n_stage2_input=int(stage2_in.shape[0]),
            preference=float(s2_pref),
            damping=float(args.damping),
            lamb=float(args.lamb),
        )
    else:
        result = hierarchical_ap_prototypes(
            slide_patches,
            preference=args.preference,
            stage2_preference=args.stage2_preference,
            target_L=args.target_L,
            stage2_method=args.stage2_method,
            damping=args.damping,
            lamb=args.lamb,
            seed=args.seed,
            min_patches_per_slide=args.min_patches_per_slide,
        )
        if args.cache_stage2_pool:
            pool_path = Path(args.cache_stage2_pool)
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            # Rebuild pool from stage1 results for exact cache
            pooled = [s.exemplars for s in result.stage1]
            stage2_in = np.concatenate(pooled, axis=0)
            np.save(pool_path, stage2_in)
            print(f"[init_prototypes_ap] cached stage2 pool → {pool_path}  n={stage2_in.shape[0]}")

    L, D = result.centers.shape
    print(
        f"[init_prototypes_ap] stage1 slides={len(result.stage1)}  "
        f"stage2_in={result.n_stage2_input}  L={L}  D={D}"
    )
    ex_counts = [s.n_exemplars for s in result.stage1]
    if ex_counts:
        print(
            f"[init_prototypes_ap] stage1 exemplars/slide: "
            f"mean={np.mean(ex_counts):.1f}  min={min(ex_counts)}  max={max(ex_counts)}"
        )

    save_hierarchical_ap_checkpoint(
        args.output,
        result,
        slide_ids=list(slide_patches.keys()) if slide_patches else slide_ids,
        patches_per_slide=args.patches_per_slide,
        seed=args.seed,
        extra={
            "split": split_note,
            "stage1_preference": args.preference,
            "target_L": args.target_L,
            "stage2_method": args.stage2_method,
        },
    )
    print(
        f"[init_prototypes_ap] wrote {args.output}  "
        f"(L={L}, stage1=ap, stage2={args.stage2_method})"
    )


if __name__ == "__main__":
    main()

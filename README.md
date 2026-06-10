# Prototype-guided Multi-head MIL for HER2 classification (HEROHE)

Code accompanying our study on H&E-only HER2 classification on the
[HEROHE](https://ecdp2020.grand-challenge.org/) cohort. The model is a
**prototype-guided multi-head multiple-instance learning (MIL)** network that
decomposes each slide into morphological *phenotype tokens* (real-patch medoid /
affinity-propagation prototypes) and aggregates them with a multi-head readout.
We evaluate it on binary HER2 (negative vs. positive) and three-class HER2
stratification (negative / low / positive), against ABMIL, CLAM, TransMIL and
DeepAttnMISL baselines under a unified 5-fold CV + test-ensemble protocol.

This repository contains the **code only**. Whole-slide images, extracted
features, and trained checkpoints are not distributed (see *Data* below).

---

## Repository layout

```
.
├── requirements.txt              # pinned Python dependencies
├── setup_vendor.sh               # clones the baseline repos at pinned commits
├── README.md
└── herohe/gp2/
    ├── configs/                  # preprocessing (segmentation/patching) configs
    ├── models/                   # our model + baseline model definitions
    │   ├── phenotype_mil_binary.py   # prototype-guided multi-head MIL (binary)
    │   ├── phenotype_mil.py          # multi-class variant
    │   ├── abmil.py / clam_mb.py / transmil.py
    │   ├── dataset.py / losses.py
    ├── prototype_discovery/      # affinity-propagation prototype construction
    ├── vendor/                   # thin adapters around upstream baselines
    │   ├── adapters/             # abmil / clam / transmil / attnmisl wrappers
    │   ├── factory.py, paths.py
    ├── results/
    │   └── all_metrics.json      # released metrics for the reported tables
    └── scripts/                  # pipeline + canonical run drivers
```

The `herohe/gp2/` package path is intentional: modules import each other as
`herohe.gp2.<...>`. Run everything **from the repository root** so these imports
resolve.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

`openslide-python` needs the native OpenSlide library:

```bash
# macOS
brew install openslide
# Ubuntu / Debian
sudo apt-get install -y openslide-tools
```

Then fetch the external baseline repositories (pinned to the exact commits used
for the reported results) and the TRIDENT feature extractor:

```bash
bash setup_vendor.sh
```

This populates `herohe/gp2/vendor/{AttentionDeepMIL,TransMIL,DeepAttnMISL}`,
`./CLAM-master`, and `./TRIDENT`. These directories are git-ignored.

---

## Data

We do **not** redistribute the HEROHE images or labels.

1. Request the HEROHE WSIs and ground-truth labels from the challenge
   organizers or you can find it on the Google Drive shared by the organizers as well. Place the MIRAX slides where the run scripts expect them
   (`herohe/wsi_test/` for the official 150-slide test set; training case folders
   at the repository root) and the label files alongside (e.g.
   `herohe/Training (ground truth).csv`, `herohe/Test (ground truth).xlsx`).
2. The run scripts reference these paths through a `REPO` root that is
   auto-detected from the script location; override any path with environment
   variables (`REPO=`, `PY=`, `WSIDIR=`, `GC_TEST=`) as needed.

---

## Reproducing the pipeline

All steps below have a ready-made driver in `herohe/gp2/scripts/`. Each driver
encodes the exact hyperparameters used for the paper; pass `--help` to the
underlying `.py` entry points to see every flag.

**1. Patch features (TRIDENT → Virchow2, 20× / 256 px).**

```bash
bash herohe/gp2/scripts/run_trident_full.sh        # training cohort
bash herohe/gp2/scripts/run_trident_test_150.sh    # official 150-slide test set
# ResNet-50 encoder ablation:
bash herohe/gp2/scripts/run_trident_resnet50_feat.sh
```

**2. Prototype construction (affinity propagation on training-fold patches).**

```bash
python herohe/gp2/scripts/init_prototypes_ap.py \
    --features_dir <features_virchow2> --folds_csv herohe/gp2/data/folds_v1.csv \
    --val_fold 0 --output herohe/gp2/data/prototypes_ap_fold0_train.pt
```

**3. Train + test-ensemble eval (5-fold CV).**

```bash
# Our model (hard-partition routing, token-level ABMIL readout) — primary config
bash herohe/gp2/scripts/run_khead_token_abmil_hard_partition_ent0.sh all
bash herohe/gp2/scripts/run_hard_partition_5fold_all.sh        # binary + 3-class

# MIL baselines (binary + three-class)
bash herohe/gp2/scripts/run_binary_baselines_5fold_s42.sh all       # ABMIL/CLAM/TransMIL
bash herohe/gp2/scripts/run_attnmisl_binary_5fold_s42.sh all
bash herohe/gp2/scripts/run_threeclass_5fold_s42.sh all
bash herohe/gp2/scripts/run_attnmisl_threeclass_5fold_s42.sh all
```

**4. Ablations.**

```bash
bash herohe/gp2/scripts/run_khead_pool_ablation.sh                       # readout (mean/concat/ABMIL)
bash herohe/gp2/scripts/run_khead_hard_partition_token_abmil_k_ablation.sh   # number of prototypes
bash herohe/gp2/scripts/run_resnet50_binary_5fold_s42.sh all             # encoder ablation
bash herohe/gp2/scripts/run_khead_token_abmil_hard_partition_ent0_resnet50.sh all
```

**5. Metrics, uncertainty, and figures.**

```bash
python herohe/gp2/scripts/compute_uncertainty.py     # bootstrap CIs, DeLong, ECE, Brier
python herohe/gp2/scripts/recompute_report_stats.py
bash   herohe/gp2/scripts/run_architecture_figures.sh
```

The metrics behind the reported tables are checked in at
`herohe/gp2/results/all_metrics.json`.

---

## Baselines and provenance

`herohe/gp2/vendor/` keeps only our thin adapters; the upstream model code is
fetched by `setup_vendor.sh` at these pinned commits:

| Baseline | Upstream | Commit |
|----------|----------|--------|
| ABMIL | [AMLab-Amsterdam/AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) | `eb0434ba2795711a45d693d60120ae53532b1b93` |
| CLAM | [mahmoodlab/CLAM](https://github.com/mahmoodlab/CLAM) | `53e2409d4a8189c682c173382964a85f114f923c` |
| TransMIL | [szc19990412/TransMIL](https://github.com/szc19990412/TransMIL) | `9d6aee57c7c72375fb9132dc58cd8c9b0f0a949c` |
| DeepAttnMISL | [uta-smile/DeepAttnMISL](https://github.com/uta-smile/DeepAttnMISL) | `d7099ed88c452aec37f16fc8e93cc0c068794c2a` |

Each upstream repository retains its own license; refer to the cloned
directories after running `setup_vendor.sh`.

---

## Notes

- The pipeline was developed and run on Apple Silicon (`--device mps`); pass
  `--device cuda` or `--device cpu` to the training/eval scripts as appropriate.
- Driver scripts auto-detect the repository root and default to the `python` on
  your `PATH`; override with `REPO=`, `PY=`, etc.

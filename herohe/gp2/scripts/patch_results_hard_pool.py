#!/usr/bin/env python3
"""Insert hard-partition mean/concat metrics into report_by_me/results.tex."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "herohe/gp2/report/report_by_me/results.tex"
MB = REPO / "herohe/gp2/runs/medoid_benchmark"
UNC = REPO / "herohe/gp2/runs/uncertainty/uncertainty.json"


def load_metrics(name: str) -> dict:
    p = MB / name / "test_eval" / f"metrics_{name}_5fold.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def rnd(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def main() -> None:
    mean_m = load_metrics("bin_hard_mean_L8")
    concat_m = load_metrics("bin_hard_concat_L8")
    mean_auc = rnd(mean_m["AUC"])
    mean_f1 = rnd(mean_m["macro_f1"])
    mean_bacc = rnd(mean_m["bACC"])
    concat_auc = rnd(concat_m["AUC"])
    concat_f1 = rnd(concat_m["macro_f1"])
    concat_bacc = rnd(concat_m["bACC"])

    text = RESULTS.read_text()

    # Update routing table rows
    text = re.sub(
        r"hard partition & mean pool & \\emph\{pending\} & --- & --- \\\\",
        f"hard partition & mean pool & {mean_auc} & {mean_f1} & {mean_bacc} \\\\",
        text,
    )
    text = re.sub(
        r"hard partition & concat & \\emph\{pending\} & --- & --- \\\\",
        f"hard partition & concat & {concat_auc} & {concat_f1} & {concat_bacc} \\\\",
        text,
    )

    # Clean caption pending note
    text = text.replace(
        "Rows marked\n  \\emph{pending} use the primary training recipe but were still running at report drafting;\n  values will be inserted when complete.",
        "Hard-partition mean and concat readouts use the same training recipe as the primary model;",
    )
    if "Hard-partition mean and concat readouts" not in text:
        text = text.replace(
            "  \\emph{pending} use the primary training recipe but were still running at report drafting;\n  values will be inserted when complete.",
            "Hard-partition mean and concat readouts use the same training recipe as the primary model;",
        )

    # Optional: add one sentence in readout ablation if still mentions pending
    pending_line = "Rows for hard\npartition with mean or concat readout were still running at report drafting and will be filled\nwhen complete."
    if pending_line.replace("\n", " ") in text.replace("\n", " "):
        text = text.replace(
            "Rows for hard\npartition with mean or concat readout were still running at report drafting and will be filled\nwhen complete.",
            f"Under hard partition, mean pooling reaches ensemble AUC {mean_auc} and concat {concat_auc}, "
            f"compared with token ABMIL at 0.826 (Table~\\ref{{tab:routing-readout}}).",
        )

    RESULTS.write_text(text)
    print(f"Patched {RESULTS}")
    print(f"  hard+mean:  AUC={mean_auc} mF1={mean_f1} bACC={mean_bacc}")
    print(f"  hard+concat: AUC={concat_auc} mF1={concat_f1} bACC={concat_bacc}")


if __name__ == "__main__":
    main()

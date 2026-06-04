"""
Model performance visualization across all evaluation runs.

(한글 요약) 모든 평가 run의 메트릭(AUROC 등)을 모아 비교 그래프로 시각화한다.
ai/ 디렉터리에서 실행한다.

Run from the ai/ directory:
    python visualize_metrics.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from drscreen.settings import find_classification_metrics_path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "runs" / "02_domain_generalization" / "_reports" / "performance_overview.png"

# --- version metadata -----------------------------------------------------------
# (file_stem_suffix, display_label, sprint, is_external, is_deployment_best)
VERSION_META = [
    ("v4.1",            "v4.1\n(SSL FT)",       1, True,  False),
    ("v6_alpha_only",   "v6\n(Focal α)",         1, True,  False),
    ("v7_messidor_train","v7\n(+Messidor)",       2, True,  False),
    ("v8_mixstyle",     "v8\n(MixStyle)",         2, True,  False),
    ("v9_fda",          "v9_fda\n(FDA)",          2, True,  True),
    ("v10_swad",        "v10_swad\n(SWAD*)",      2, True,  False),
    ("v11_fda_swad",    "v11\n(FDA+SWAD)",        2, True,  False),
    ("v12_fda_imagenet","v12\n(FDA+IN)",          2, True,  False),
    ("v13_fda_swad",    "v13\n(FDA+SWAD v2)",     2, True,  False),
    ("v14_ibn",         "v14\n(IBN)",             2, True,  False),
]

SPRINT2_KEYS = {m[0] for m in VERSION_META if m[2] == 2}


def load_metrics(version_key: str, external: bool) -> dict | None:
    prefix = "external_test" if external else "test"
    path = find_classification_metrics_path(PROJECT_ROOT, version_key, split_name=prefix)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_metric(data: dict, key: str, optimal: bool = False) -> float | None:
    if data is None:
        return None
    src = data.get("metrics_at_optimal_threshold" if optimal else "metrics", {})
    return src.get(key)


def main() -> None:
    records = []
    for version_key, label, sprint, external, is_best in VERSION_META:
        data = load_metrics(version_key, external)
        if data is None:
            continue
        records.append({
            "key": version_key,
            "label": label,
            "sprint": sprint,
            "is_best": is_best,
            "auroc": get_metric(data, "auroc"),
            "opt_thr": data.get("optimal_threshold"),
            "sensitivity": get_metric(data, "sensitivity", optimal=True),
            "specificity": get_metric(data, "specificity", optimal=True),
            "f1": get_metric(data, "f1", optimal=True),
        })

    if not records:
        print("No evaluation files found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "DR Screening AI — Performance Overview (External Test: DDR n=12,522 / Messidor for v4.1–v6)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    colors = {1: "#4C72B0", 2: "#DD8452"}
    star_color = "#2ca02c"

    # ── Panel 1: AUROC bar chart ────────────────────────────────────────────────
    ax1 = axes[0]
    xs = np.arange(len(records))
    bar_colors = [star_color if r["is_best"] else colors[r["sprint"]] for r in records]
    aurocs = [r["auroc"] for r in records]
    bars = ax1.bar(xs, aurocs, color=bar_colors, edgecolor="white", linewidth=0.5)

    ax1.set_ylim(0.60, 0.95)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([r["label"] for r in records], fontsize=7.5)
    ax1.set_ylabel("AUROC")
    ax1.set_title("AUROC by Version")
    ax1.axhline(0.90, color="red", linestyle="--", linewidth=0.8, label="Clinical target (0.90)")
    ax1.axhline(0.80, color="orange", linestyle=":", linewidth=0.8, label="Acceptable (0.80)")

    for bar, val in zip(bars, aurocs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=6.5, rotation=45,
        )

    legend_patches = [
        mpatches.Patch(color=colors[1], label="Sprint 1"),
        mpatches.Patch(color=colors[2], label="Sprint 2"),
        mpatches.Patch(color=star_color, label="Deployment best"),
    ]
    ax1.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="red",    linestyle="--", linewidth=0.8, label="Clinical target"),
        plt.Line2D([0], [0], color="orange", linestyle=":",  linewidth=0.8, label="Acceptable"),
    ], fontsize=7, loc="lower right")

    # ── Panel 2: AUROC vs Optimal Threshold (Sprint 2 scatter) ─────────────────
    ax2 = axes[1]
    for r in records:
        if r["sprint"] != 2:
            continue
        c = star_color if r["is_best"] else colors[2]
        ax2.scatter(r["opt_thr"], r["auroc"], color=c, s=80, zorder=3)
        ax2.annotate(
            r["label"].replace("\n", " "),
            (r["opt_thr"], r["auroc"]),
            textcoords="offset points", xytext=(6, 2),
            fontsize=7,
        )

    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Ideal threshold (0.5)")
    ax2.set_xlabel("Optimal Threshold")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Sprint 2 — AUROC vs Optimal Threshold\n(right = better calibrated)")
    ax2.set_xlim(-0.02, 0.55)
    ax2.legend(fontsize=7)
    ax2.annotate(
        "← threshold bias",
        xy=(0.05, 0.836), fontsize=8, color="gray", style="italic",
    )

    # ── Panel 3: Sensitivity / Specificity at optimal threshold ────────────────
    ax3 = axes[2]
    s2_records = [r for r in records if r["sprint"] == 2]
    x3 = np.arange(len(s2_records))
    width = 0.35

    sens = [r["sensitivity"] for r in s2_records]
    spec = [r["specificity"] for r in s2_records]

    b1 = ax3.bar(x3 - width / 2, sens, width, label="Sensitivity", color="#4878d0", alpha=0.85)
    b2 = ax3.bar(x3 + width / 2, spec, width, label="Specificity", color="#ee854a", alpha=0.85)

    ax3.axhline(0.90, color="red", linestyle="--", linewidth=0.8, label="Clinical sensitivity target")
    ax3.set_ylim(0.55, 1.05)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([r["label"] for r in s2_records], fontsize=7.5)
    ax3.set_ylabel("Score @ Optimal Threshold")
    ax3.set_title("Sprint 2 — Sensitivity & Specificity\n(at optimal threshold)")
    ax3.legend(fontsize=7)

    for bar, val in zip(list(b1) + list(b2), sens + spec):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=6,
        )

    # deployment best marker
    for i, r in enumerate(s2_records):
        if r["is_best"]:
            ax3.annotate(
                "★ deploy",
                xy=(i, 0.56), ha="center", fontsize=7, color=star_color, fontweight="bold",
            )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

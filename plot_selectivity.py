"""
plot_selectivity.py
===================
Generates Latency & Recall vs Selectivity plots for the main 200k benchmark.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config


def get_latest_results_dir() -> Path:
    dirs = [d for d in config.RESULTS_DIR.iterdir()
            if d.is_dir() and d.name.startswith("main_200k_")]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def generate_selectivity_plots():
    res_dir = get_latest_results_dir()
    if not res_dir:
        print("Error: No main benchmark results found.")
        return

    csv_path = res_dir / "phase_1_selectivity.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found in {res_dir}")
        return

    print(f"Found results in {res_dir.name}")

    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })

    colors = {
        "cbo": "#2196F3",
        "pf": "#4CAF50",
        "idsel": "#9C27B0",
        "bf": "#FF5722",
    }

    fig, (ax_lat, ax_rec) = plt.subplots(1, 2, figsize=(16, 5))
    
    df = pd.read_csv(csv_path).sort_values("selectivity_pct")
    n_docs = df["n_docs"].iloc[0]
    title_suffix = f"{n_docs // 1000}k docs"

    sel = df["selectivity_pct"]

    # ── Latency ──────────────────────────────────────────────────
    ax_lat.plot(sel, df["bitmap_bf_latency_ms"], "s-",
                color=colors["bf"], linewidth=1.5, alpha=0.5,
                markersize=4, label="Bitmap (BF)")
    ax_lat.plot(sel, df["idsel_latency_ms"], "D-",
                color=colors["idsel"], linewidth=1.5, alpha=0.6,
                markersize=4, label="IDSelector")
    ax_lat.plot(sel, df["pf_latency_ms"], "^-",
                color=colors["pf"], linewidth=1.5, alpha=0.6,
                markersize=4, label="PostFilter")
    ax_lat.plot(sel, df["cbo_latency_ms"], "o-",
                color=colors["cbo"], linewidth=2.5, markersize=6,
                label="CBO", zorder=5)

    ax_lat.axhline(config.CBO_L_MAX, color="red", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"L_MAX ({config.CBO_L_MAX}ms)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_xlabel("Selectivity (%)")
    ax_lat.set_title(f"Latency vs Selectivity — {title_suffix}",
                     fontweight="bold")
    ax_lat.set_ylim(0, min(150, df["bitmap_bf_latency_ms"].max() * 1.1))
    ax_lat.legend(fontsize=8, loc="upper left")

    # ── Recall ───────────────────────────────────────────────────
    ax_rec.plot(sel, df["bitmap_bf_recall"] * 100, "s-",
                color=colors["bf"], linewidth=1.5, alpha=0.5,
                markersize=4, label="Bitmap (BF)")
    ax_rec.plot(sel, df["idsel_recall"] * 100, "D-",
                color=colors["idsel"], linewidth=1.5, alpha=0.6,
                markersize=4, label="IDSelector")
    ax_rec.plot(sel, df["pf_recall"] * 100, "^-",
                color=colors["pf"], linewidth=1.5, alpha=0.6,
                markersize=4, label="PostFilter")
    ax_rec.plot(sel, df["cbo_recall"] * 100, "o-",
                color=colors["cbo"], linewidth=2.5, markersize=6,
                label="CBO", zorder=5)

    ax_rec.axhline(config.CBO_R_TARGET * 100, color="red",
                   linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"R_TARGET ({int(config.CBO_R_TARGET*100)}%)")
    ax_rec.set_ylabel("Recall (%)")
    ax_rec.set_xlabel("Selectivity (%)")
    ax_rec.set_title(f"Recall vs Selectivity — {title_suffix}",
                     fontweight="bold")
    ax_rec.set_ylim(60, 102)
    ax_rec.legend(fontsize=8, loc="lower right")

    fig.tight_layout()

    out_path = res_dir / "selectivity_plot.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"✓ Plot saved → {out_path}")


if __name__ == "__main__":
    generate_selectivity_plots()

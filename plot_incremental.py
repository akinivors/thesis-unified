import json
import matplotlib.pyplot as plt
from pathlib import Path
import config

def get_latest_incremental_dir() -> Path:
    dirs = [d for d in config.RESULTS_DIR.iterdir()
            if d.is_dir() and d.name.startswith("incremental_freeze_")]
    if not dirs:
        return None
    return sorted(dirs)[-1]

def generate_incremental_plot():
    res_dir = get_latest_incremental_dir()
    if not res_dir:
        print("Error: No incremental benchmark results found.")
        return

    json_path = res_dir / "crossover_evolution.json"
    if not json_path.exists():
        print(f"Error: {json_path} not found in {res_dir}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter out None values for plotting or treat None as 0
    phases = []
    docs = []
    crossovers = []
    for d in data:
        phases.append(d["phase"])
        docs.append(f"{d['n_docs'] // 1000}k")
        val = d["crossover_selectivity"]
        crossovers.append(val if val is not None else 0.0)

    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
    })

    plt.figure(figsize=(8, 5))
    
    # Plot line
    plt.plot(docs, crossovers, marker='o', linewidth=2.5, markersize=8, color="#E91E63")
    
    # Fill area under curve
    plt.fill_between(docs, crossovers, alpha=0.1, color="#E91E63")

    plt.title("Crossover Selectivity vs Corpus Size", pad=15, fontweight="bold")
    plt.xlabel("Corpus Size (Number of Documents)")
    plt.ylabel("Crossover Selectivity (Brute Force wins below)")
    plt.ylim(0, max(crossovers) * 1.2 if max(crossovers) > 0 else 0.1)
    
    # Add data labels
    for i, txt in enumerate(crossovers):
        if txt == 0.0:
            label = "None (No Crossover)"
        else:
            label = f"{txt:.3f}"
        plt.annotate(label, (docs[i], crossovers[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontweight='bold')

    plt.tight_layout()
    out_path = res_dir / "crossover_evolution_plot.png"
    plt.savefig(out_path)
    print(f"✓ Plot saved → {out_path}")

if __name__ == "__main__":
    generate_incremental_plot()

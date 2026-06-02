"""
build_index.py — One-Time Index Builder for the Unified System
==============================================================
Converts your existing bandit-cbo data files into the directory layout
that the unified system (FAISSIndex.load / main_benchmark.py) expects.

What it does
------------
1. Loads embeddings from the old system's  electronics_vectors.npy
2. Loads metadata from the old system's    metadata_671k.csv
3. Caps at MAX_DOCUMENTS (config.MAX_DOCUMENTS, default 400 000)
4. L2-normalises embeddings (required for inner-product == cosine)
5. Builds a fresh FAISS IndexHNSWFlat (METRIC_INNER_PRODUCT)
6. Saves to  <thesis-unified>/index_data/  as:
       hnsw.index          FAISS binary index
       embeddings.npy      normalised float32 array  (N × dim)
       id_mappings.pkl     {int_to_str, str_to_int} dicts
       all_metadatas.pkl   list of metadata dicts

Run once, then leave it — every benchmark script reads from index_data/.

Usage
-----
    python build_index.py                          # uses default paths
    python build_index.py --embeddings /path/to/electronics_vectors.npy \\
                          --metadata   /path/to/metadata_671k.csv

Fields in all_metadatas.pkl
----------------------------
The old metadata CSV has:  idx, asin, brand, main_cat, price, year, title
The unified bitmap index uses: main_cat, brand, overall, verified

  overall  — not in old CSV; filled with 0.0 (sentinel "unknown").
              generate_filters() will skip overall-based filters since
              selectivity is always 0 with this sentinel.
  verified — not in old CSV; filled with False similarly.

If you later want real overall/verified data, join your Electronics.json
reviews against asin before running this script, or re-process from raw.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# ── Make sure project root is importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config
from faiss_index import FAISSIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Default source paths (old bandit-cbo system) ──────────────────────────────
# Adjust these if your data lives somewhere else, or use --embeddings / --metadata
_OLD_SYSTEM_DIR = Path(__file__).parent.parent / "bandit_cbo" / "data"
DEFAULT_EMB_PATH  = _OLD_SYSTEM_DIR / "electronics_vectors.npy"
DEFAULT_META_PATH = _OLD_SYSTEM_DIR / "metadata_671k.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified FAISS index from old-system data.")
    p.add_argument(
        "--embeddings", type=Path, default=DEFAULT_EMB_PATH,
        help=f"Path to electronics_vectors.npy  (default: {DEFAULT_EMB_PATH})",
    )
    p.add_argument(
        "--metadata", type=Path, default=DEFAULT_META_PATH,
        help=f"Path to metadata_671k.csv  (default: {DEFAULT_META_PATH})",
    )
    p.add_argument(
        "--out", type=Path, default=config.INDEX_DIR,
        help=f"Output directory  (default: {config.INDEX_DIR})",
    )
    p.add_argument(
        "--max-docs", type=int, default=config.MAX_DOCUMENTS,
        help=f"Maximum documents to index  (default: {config.MAX_DOCUMENTS})",
    )
    return p.parse_args()


def load_metadata(csv_path: Path, max_docs: int) -> list[dict]:
    """Load the old CSV and convert rows to metadata dicts.

    Fields in the old CSV: idx, asin, brand, main_cat, price, year, title
    We fill in overall=0.0 and verified=False as sentinels for missing fields.
    """
    log.info("Loading metadata from %s (capped at %d rows) …", csv_path, max_docs)

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_docs:
                break
            rows.append({
                "asin":     row.get("asin", ""),
                "brand":    row.get("brand", "Unknown") or "Unknown",
                "main_cat": row.get("main_cat", "Unknown") or "Unknown",
                "title":    row.get("title", ""),
                # overall and verified are not in the old CSV.
                # sentinel 0.0 means "unknown" — generate_filters() will
                # skip overall-based filters because selectivity comes out 0.
                "overall":  0.0,
                "verified": False,
            })

    log.info("Loaded %d metadata rows.", len(rows))
    return rows


def main() -> None:
    args = parse_args()

    # ── Validate source files ─────────────────────────────────────────────────
    if not args.embeddings.exists():
        log.error(
            "Embeddings file not found: %s\n"
            "Pass the correct path with --embeddings /path/to/electronics_vectors.npy",
            args.embeddings,
        )
        sys.exit(1)

    if not args.metadata.exists():
        log.error(
            "Metadata CSV not found: %s\n"
            "Pass the correct path with --metadata /path/to/metadata_671k.csv",
            args.metadata,
        )
        sys.exit(1)

    # ── Load embeddings ───────────────────────────────────────────────────────
    log.info("Loading embeddings from %s …", args.embeddings)
    t0 = time.perf_counter()
    embeddings_full = np.load(args.embeddings)
    log.info(
        "Embeddings loaded: shape=%s, dtype=%s  (%.1fs)",
        embeddings_full.shape, embeddings_full.dtype,
        time.perf_counter() - t0,
    )

    n_docs = min(args.max_docs, len(embeddings_full))
    embeddings = embeddings_full[:n_docs].astype(np.float32)
    del embeddings_full  # free memory

    log.info("Using first %d documents (max_docs=%d).", n_docs, args.max_docs)

    # ── Load metadata ─────────────────────────────────────────────────────────
    all_metadatas = load_metadata(args.metadata, n_docs)

    if len(all_metadatas) != n_docs:
        log.error(
            "Embedding count (%d) != metadata count (%d). "
            "Make sure both files were generated from the same pipeline run.",
            n_docs, len(all_metadatas),
        )
        sys.exit(1)

    # ── Build FAISSIndex ──────────────────────────────────────────────────────
    log.info("Building FAISSIndex (HNSW M=%d, efConstruction=%d) …",
             config.HNSW_M, config.HNSW_EF_CONSTR)
    log.info("This takes 5–20 minutes for 400 000 documents. Please wait.")

    doc_ids = [str(i) for i in range(n_docs)]

    t1 = time.perf_counter()
    faiss_idx = FAISSIndex(dim=embeddings.shape[1])
    faiss_idx.build(embeddings, doc_ids)   # normalises in-place & builds HNSW
    elapsed = time.perf_counter() - t1
    log.info("HNSW index built: ntotal=%d  (%.1fs = %.1f min)",
             faiss_idx.hnsw_index.ntotal, elapsed, elapsed / 60)

    # ── Save all_metadatas.pkl (needed by main_benchmark.py) ─────────────────
    args.out.mkdir(parents=True, exist_ok=True)

    meta_path = args.out / "all_metadatas.pkl"
    log.info("Saving metadata → %s …", meta_path)
    with open(meta_path, "wb") as fh:
        pickle.dump(all_metadatas, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Save FAISS index + embeddings + id_mappings ───────────────────────────
    log.info("Saving FAISSIndex → %s …", args.out)
    faiss_idx.save(args.out)

    # ── Quick sanity check ────────────────────────────────────────────────────
    log.info("Running sanity search …")
    q = faiss_idx._embeddings[0:1].copy()
    dists, ids = faiss_idx.hnsw_index.search(q, 5)
    assert ids[0, 0] == 0, f"Top result should be the query itself (got {ids[0,0]})"
    log.info("Sanity check passed: top-1 for doc-0 is doc-0 (dist=%.4f).", dists[0, 0])

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Index build complete.")
    log.info("  Documents indexed : %d", n_docs)
    log.info("  Vector dim        : %d", faiss_idx.dim)
    log.info("  Output directory  : %s", args.out)
    log.info("")
    log.info("Files written:")
    for fname in ["hnsw.index", "embeddings.npy", "id_mappings.pkl", "all_metadatas.pkl"]:
        p = args.out / fname
        size_mb = p.stat().st_size / 1024 / 1024 if p.exists() else 0
        log.info("  %-25s  %.1f MB", fname, size_mb)
    log.info("=" * 60)
    log.info("")
    log.info("You can now run:  python main_benchmark.py")


if __name__ == "__main__":
    main()

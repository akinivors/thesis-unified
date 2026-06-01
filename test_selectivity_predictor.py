"""
test_selectivity_predictor.py
=============================
Tests the MiniBitmapPredictor accuracy & speed against brute-force
(full bitmap) selectivity. No vector search or query execution needed.

Outputs:
  1. Per-filter comparison table (CSV): true vs estimated selectivity
  2. Aggregate accuracy metrics (MAE, MAPE, Max Error)
  3. Timing statistics (µs per estimation)
  4. Accuracy breakdown by selectivity range
  5. Accuracy breakdown by filter type
"""

import csv
import json
import logging
import pickle
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

import config
from bitmap_index import BitmapIndex
from filters import FilterSpec, generate_filters, _build_candidate_pool
from selectivity_predictor import MiniBitmapPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Result Data Classes ──────────────────────────────────────────────────────

@dataclass
class FilterResult:
    filter_name: str
    filter_type: str          # "single", "compound", "range", "in_set"
    true_selectivity: float
    estimated_selectivity: float
    absolute_error: float
    relative_error_pct: float
    estimation_time_us: float


@dataclass
class AggregateMetrics:
    n_filters: int
    mae: float                # Mean Absolute Error
    mape_pct: float           # Mean Absolute Percentage Error
    max_abs_error: float
    median_abs_error: float
    mean_estimation_us: float
    median_estimation_us: float
    p99_estimation_us: float


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_filter(name: str) -> str:
    """Classify a filter name into a type category."""
    if " AND " in name:
        return "compound"
    if " IN " in name:
        return "in_set"
    if ">=" in name or "<=" in name:
        return "range"
    return "single"


def selectivity_bucket(sel: float) -> str:
    """Return a human-readable selectivity range label."""
    if sel < 0.10:
        return "0-10%"
    elif sel < 0.20:
        return "10-20%"
    elif sel < 0.30:
        return "20-30%"
    elif sel < 0.50:
        return "30-50%"
    elif sel < 0.70:
        return "50-70%"
    else:
        return "70-100%"


# ── Main Test ────────────────────────────────────────────────────────────────

def run_test(sample_ratios: list[float] = None):
    if sample_ratios is None:
        sample_ratios = [0.01, 0.03, 0.05, 0.10]

    out_dir = config.RESULTS_DIR / f"selectivity_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────
    logger.info("Loading metadata from %s ...", config.INDEX_DIR)
    with open(config.INDEX_DIR / "all_metadatas.pkl", "rb") as fh:
        all_metadatas = pickle.load(fh)

    N = len(all_metadatas)
    logger.info("Loaded %d documents", N)

    # ── Build full bitmap (ground truth) ─────────────────────────────────
    logger.info("Building full BitmapIndex for ground truth...")
    t0 = time.time()
    full_bitmap = BitmapIndex()
    for i, doc in enumerate(all_metadatas):
        full_bitmap.add_document(i, str(i), doc)
    full_build_time = time.time() - t0
    logger.info("Full BitmapIndex built in %.2fs (%d docs)", full_build_time, N)

    # ── Generate ALL candidate filters (not just targeted ones) ──────────
    logger.info("Generating candidate filter pool...")
    all_candidates = _build_candidate_pool(all_metadatas)

    # Compute true selectivity for each via full bitmap
    for f in all_candidates:
        matching = f.resolve_bitmap(full_bitmap)
        f.actual_selectivity = len(matching) / N

    # Filter out near-zero selectivity (degenerate filters)
    candidates = [f for f in all_candidates if f.actual_selectivity > 0.001]
    logger.info("Using %d filters for testing (filtered from %d)", len(candidates), len(all_candidates))

    # ── Test each sample ratio ───────────────────────────────────────────
    all_ratio_results = {}

    for ratio in sample_ratios:
        logger.info("\n%s", "=" * 60)
        logger.info("Testing sample_ratio = %.0f%% (%d docs)",
                     ratio * 100, int(N * ratio))
        logger.info("=" * 60)

        # Build predictor
        t0 = time.time()
        predictor = MiniBitmapPredictor(all_metadatas, sample_ratio=ratio, seed=42)
        build_time_ms = (time.time() - t0) * 1000
        logger.info("Predictor built in %.1fms", build_time_ms)

        # Run estimation on all filters
        results: List[FilterResult] = []

        for fspec in candidates:
            true_sel = fspec.actual_selectivity

            # Run estimation multiple times for stable timing
            timings = []
            est_sel = 0.0
            for _ in range(5):
                est, us = predictor.estimate_timed(fspec)
                timings.append(us)
                est_sel = est

            abs_err = abs(true_sel - est_sel)
            rel_err = (abs_err / true_sel * 100) if true_sel > 0.001 else 0.0

            results.append(FilterResult(
                filter_name=fspec.name,
                filter_type=classify_filter(fspec.name),
                true_selectivity=round(true_sel, 6),
                estimated_selectivity=round(est_sel, 6),
                absolute_error=round(abs_err, 6),
                relative_error_pct=round(rel_err, 2),
                estimation_time_us=round(np.median(timings), 2),
            ))

        # ── Aggregate metrics ────────────────────────────────────────────
        abs_errors = [r.absolute_error for r in results]
        rel_errors = [r.relative_error_pct for r in results if r.true_selectivity > 0.01]
        timings_all = [r.estimation_time_us for r in results]

        agg = AggregateMetrics(
            n_filters=len(results),
            mae=round(np.mean(abs_errors), 6),
            mape_pct=round(np.mean(rel_errors), 2) if rel_errors else 0.0,
            max_abs_error=round(max(abs_errors), 6),
            median_abs_error=round(np.median(abs_errors), 6),
            mean_estimation_us=round(np.mean(timings_all), 2),
            median_estimation_us=round(np.median(timings_all), 2),
            p99_estimation_us=round(np.percentile(timings_all, 99), 2),
        )

        logger.info("\n--- Aggregate Metrics (ratio=%.0f%%) ---", ratio * 100)
        logger.info("  Filters tested     : %d", agg.n_filters)
        logger.info("  MAE                : %.4f (%.2f puan)", agg.mae, agg.mae * 100)
        logger.info("  MAPE               : %.2f%%", agg.mape_pct)
        logger.info("  Max Absolute Error : %.4f (%.2f puan)", agg.max_abs_error, agg.max_abs_error * 100)
        logger.info("  Median Abs Error   : %.4f (%.2f puan)", agg.median_abs_error, agg.median_abs_error * 100)
        logger.info("  Median Time        : %.1f µs", agg.median_estimation_us)
        logger.info("  P99 Time           : %.1f µs", agg.p99_estimation_us)

        # ── Breakdown by selectivity range ───────────────────────────────
        logger.info("\n--- Accuracy by Selectivity Range ---")
        by_range: Dict[str, list] = {}
        for r in results:
            bucket = selectivity_bucket(r.true_selectivity)
            by_range.setdefault(bucket, []).append(r.absolute_error)

        range_breakdown = {}
        for bucket in ["0-10%", "10-20%", "20-30%", "30-50%", "50-70%", "70-100%"]:
            errs = by_range.get(bucket, [])
            if errs:
                mae = np.mean(errs)
                logger.info("  %s : MAE=%.4f (%.2f puan), n=%d",
                            bucket, mae, mae * 100, len(errs))
                range_breakdown[bucket] = {
                    "mae": round(float(mae), 6),
                    "n_filters": len(errs),
                }

        # ── Breakdown by filter type ─────────────────────────────────────
        logger.info("\n--- Accuracy by Filter Type ---")
        by_type: Dict[str, list] = {}
        for r in results:
            by_type.setdefault(r.filter_type, []).append(r.absolute_error)

        type_breakdown = {}
        for ftype, errs in sorted(by_type.items()):
            mae = np.mean(errs)
            logger.info("  %-10s : MAE=%.4f (%.2f puan), n=%d",
                        ftype, mae, mae * 100, len(errs))
            type_breakdown[ftype] = {
                "mae": round(float(mae), 6),
                "n_filters": len(errs),
            }

        # ── Save per-ratio CSV ───────────────────────────────────────────
        ratio_label = f"{int(ratio * 100)}pct"
        csv_path = out_dir / f"filter_comparison_{ratio_label}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in sorted(results, key=lambda x: x.true_selectivity):
                writer.writerow(asdict(r))

        all_ratio_results[ratio_label] = {
            "sample_ratio": ratio,
            "sample_size": int(N * ratio),
            "build_time_ms": round(build_time_ms, 2),
            "aggregate": asdict(agg),
            "by_selectivity_range": range_breakdown,
            "by_filter_type": type_breakdown,
            "predictor_stats": predictor.get_stats(),
        }

    # ── Save summary JSON ────────────────────────────────────────────────
    summary_path = out_dir / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_ratio_results, f, indent=2, default=str)

    # ── needs_rebuild test ───────────────────────────────────────────────
    logger.info("\n--- needs_rebuild() Test ---")
    predictor = MiniBitmapPredictor(all_metadatas, sample_ratio=0.03)
    logger.info("  N_at_build = %d", predictor.N_at_build)
    logger.info("  needs_rebuild(N * 1.10) = %s", predictor.needs_rebuild(int(N * 1.10)))
    logger.info("  needs_rebuild(N * 1.29) = %s", predictor.needs_rebuild(int(N * 1.29)))
    logger.info("  needs_rebuild(N * 1.31) = %s", predictor.needs_rebuild(int(N * 1.31)))
    logger.info("  needs_rebuild(N * 0.69) = %s", predictor.needs_rebuild(int(N * 0.69)))
    logger.info("  needs_rebuild(N * 0.50) = %s", predictor.needs_rebuild(int(N * 0.50)))

    logger.info("\n✓ All results saved to %s", out_dir)


if __name__ == "__main__":
    run_test()

#!/usr/bin/env python3
"""
incremental_benchmark.py
========================
Validates CBO behaviour across corpus scales: 200k → 335k → 500k → 671k.

For every corpus size the script:
  1. Slices the full 671k dataset (loaded once)
  2. Builds a fresh FAISS HNSW index and BitmapIndex for that slice
  3. Applies adaptive L_MAX  (config.compute_l_max)
  4. Trains CBO until freeze  (N_WARMUP = 5 000 by default)
  5. Evaluates CBO / always-bitmap / always-post / oracle across all filters

Key research questions answered:
  • Does θ* shift predictably as corpus grows under adaptive L_MAX?
  • Does oracle gap stay near the irreducible floor at every scale?
  • Is N_WARMUP = 5 000 sufficient (or overkill) for smaller corpora?
  • Does the linear L_MAX formula preserve L_norm(bitmap) across scales?

Outputs  (results/incremental_<timestamp>/):
  crossover_evolution.json          per-phase summary: θ*, rewards, gaps
  summary_table.csv                 machine-readable cross-scale comparison
  phase_<n>_<ndocs>_summary.json    full eval_summary for each size
  phase_<n>_<ndocs>_per_filter.csv  per-filter breakdown
  phase_<n>_<ndocs>_learning.csv    learning curve (every 100 steps)

Usage:
    python incremental_benchmark.py
    python incremental_benchmark.py --n-queries 100   # ~half the runtime
    python incremental_benchmark.py --corpus-sizes 200000,671750
    python incremental_benchmark.py --n-warmup 2000   # test old setting
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec, generate_filters
from strategies import BitmapPreFilter, BruteForce, PostFilter
from selectivity_predictor import MiniBitmapPredictor
from cbo import ContextualBanditOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CORPUS_SIZES = [200_000, 335_000, 500_000, 671_750]
MAX_TRAINING_QUERIES = 15_000
SNAPSHOT_INTERVAL    = 100

SELECTIVITY_ZONES = [
    ("0-10%",  0.00, 0.10),
    ("10-20%", 0.10, 0.20),
    ("20-30%", 0.20, 0.30),
    ("30-40%", 0.30, 0.40),
    ("40-50%", 0.40, 0.50),
    ("50-90%", 0.50, 0.90),
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    filter_name: str
    filter_type: str
    selectivity_pct: float
    zone: str
    # CBO
    cbo_lat_p50: float
    cbo_lat_p95: float
    cbo_recall: float
    cbo_sla: float
    cbo_reward: float
    # Oracle
    oracle_reward: float
    oracle_gap: float
    cbo_optimal_rate: float
    # Bitmap baseline
    bm_lat_p50: float
    bm_lat_p95: float
    bm_recall: float
    bm_sla: float
    bm_reward: float
    # PostFilter baseline
    pf_lat_p50: float
    pf_lat_p95: float
    pf_recall: float
    pf_sla: float
    pf_reward: float


@dataclass
class LcPoint:
    step: int
    frozen: bool
    crossover: Optional[float]
    sla_last100: float
    reward_last100: float
    q_bm_020: float
    q_pf_020: float
    q_bm_030: float
    q_pf_030: float
    q_bm_040: float
    q_pf_040: float


# ── Helpers ────────────────────────────────────────────────────────────────────

def classify_filter(name: str) -> str:
    if " AND " in name:
        return "compound"
    if " IN " in name:
        return "in_set"
    if ">=" in name or "<=" in name:
        return "range"
    return "single"


def zone_for(sel: float) -> str:
    for label, lo, hi in SELECTIVITY_ZONES:
        if lo <= sel < hi:
            return label
    return "50-90%"


def sla(recalls: list, tgt: float = config.CBO_R_TARGET) -> float:
    return sum(r >= tgt for r in recalls) / len(recalls) if recalls else 0.0


def pct(v: list, p: float) -> float:
    return float(np.percentile(v, p)) if v else 0.0


def reward_fn(lat: float, rec: float) -> float:
    """SoftCliff reward — always reads config.CBO_L_MAX (updated per phase)."""
    margin = config.CBO_RECALL_MARGIN
    if rec < config.CBO_R_TARGET - margin:
        return 0.0
    l_norm = max(0.0, 1.0 - lat / config.CBO_L_MAX)
    if rec >= config.CBO_R_TARGET:
        return l_norm
    return l_norm * ((rec / config.CBO_R_TARGET) ** config.CBO_BETA)


# ── One-time data load ─────────────────────────────────────────────────────────

def load_base(index_dir: Path):
    """Load full 671k embeddings + metadata once; sliced per phase."""
    log.info("Loading base index from %s …", index_dir)
    full = FAISSIndex.load(index_dir)
    with open(index_dir / "all_metadatas.pkl", "rb") as fh:
        meta = pickle.load(fh)
    embeddings = full._embeddings
    log.info("Base data: %d docs, dim=%d", embeddings.shape[0], embeddings.shape[1])
    return embeddings, meta


# ── Per-phase corpus ──────────────────────────────────────────────────────────

def build_phase_corpus(
    n_docs: int,
    full_embeddings: np.ndarray,
    full_meta: list,
) -> Tuple[np.ndarray, list, FAISSIndex, BitmapIndex]:
    n_docs = min(n_docs, full_embeddings.shape[0])
    emb  = full_embeddings[:n_docs]
    meta = full_meta[:n_docs]

    log.info("Building FAISS HNSW index (%d docs) …", n_docs)
    faiss_idx = FAISSIndex(dim=emb.shape[1])
    faiss_idx.build(emb, [str(i) for i in range(n_docs)])

    log.info("Building BitmapIndex …")
    bmap = BitmapIndex()
    for i, doc in enumerate(meta):
        bmap.add_document(i, str(i), doc)

    return emb, meta, faiss_idx, bmap


# ── Cache pre-computation ─────────────────────────────────────────────────────

def precompute(
    filters: List[FilterSpec],
    q_embs: np.ndarray,
    brute: BruteForce,
    post: PostFilter,
    bmap: BitmapPreFilter,
    n_q: int,
) -> Dict:
    cache = {
        "bitmap_prefilter": [[None] * n_q for _ in range(len(filters))],
        "post_filter":      [[None] * n_q for _ in range(len(filters))],
    }
    for fi, fspec in enumerate(tqdm(filters, desc="  Pre-computing cache")):
        sel = fspec.actual_selectivity
        for qi in range(n_q):
            q = q_embs[qi].reshape(1, -1)
            ref = set(brute.search(q, config.CBO_TOP_K, fspec).ids)
            n   = len(ref) or 1

            r = post.search(q, config.CBO_TOP_K, fspec, selectivity=sel)
            cache["post_filter"][fi][qi] = (
                r.total_time_ms,
                len(set(r.ids) & ref) / n,
            )
            r = bmap.search(q, config.CBO_TOP_K, fspec, selectivity=sel)
            cache["bitmap_prefilter"][fi][qi] = (
                r.total_time_ms,
                len(set(r.ids) & ref) / n,
            )
    return cache


# ── CBO training ──────────────────────────────────────────────────────────────

def train_phase(
    optimizer: ContextualBanditOptimizer,
    predictor: MiniBitmapPredictor,
    filters: List[FilterSpec],
    cache: Dict,
    n_queries: int,
    n_warmup: int,
) -> Tuple[int, List[LcPoint]]:

    log.info(
        "  Training CBO (max %d queries, N_WARMUP=%d) …",
        MAX_TRAINING_QUERIES, n_warmup,
    )
    curve: List[LcPoint] = []
    win_sla: List[float] = []
    win_rew: List[float] = []
    frozen_at = -1

    def snap(step: int) -> LcPoint:
        def qp(sel):
            qv = optimizer.qtable.get_q_values(sel)
            return qv.get("bitmap_prefilter", 1.0), qv.get("post_filter", 1.0)
        b2, p2 = qp(0.20)
        b3, p3 = qp(0.30)
        b4, p4 = qp(0.40)
        return LcPoint(
            step=step,
            frozen=optimizer.is_frozen,
            crossover=optimizer.get_crossover_estimate(),
            sla_last100=round(sla(win_sla[-100:]), 4),
            reward_last100=round(float(np.mean(win_rew[-100:])) if win_rew else 0, 4),
            q_bm_020=round(b2, 4), q_pf_020=round(p2, 4),
            q_bm_030=round(b3, 4), q_pf_030=round(p3, 4),
            q_bm_040=round(b4, 4), q_pf_040=round(p4, 4),
        )

    for step in range(MAX_TRAINING_QUERIES):
        fspec = random.choice(filters)
        qi    = random.randrange(n_queries)
        fi    = filters.index(fspec)

        est, _ = predictor.estimate_timed(fspec)
        strat  = optimizer.route(est)
        lat, rec = cache[strat][fi][qi]
        rew = optimizer.feedback(est, strat, lat, rec)

        win_rew.append(rew)
        win_sla.append(float(rec >= config.CBO_R_TARGET))

        if (step + 1) % SNAPSHOT_INTERVAL == 0:
            curve.append(snap(step + 1))
            if (step + 1) % 500 == 0 or optimizer.is_frozen:
                log.info(
                    "    [step %5d] frozen=%-5s θ*=%-6s  sla=%.0f%%"
                    "  q_bm@0.30=%.3f  q_pf@0.30=%.3f",
                    step + 1,
                    optimizer.is_frozen,
                    f"{optimizer.frozen_crossover_point:.3f}"
                      if optimizer.frozen_crossover_point else "None",
                    win_sla[-100:].count(1.0) / min(100, len(win_sla)) * 100,
                    optimizer.qtable.get_q_values(0.30).get("bitmap_prefilter", 1.0),
                    optimizer.qtable.get_q_values(0.30).get("post_filter", 1.0),
                )

        if optimizer.is_frozen and frozen_at == -1:
            frozen_at = step + 1
            log.info("  CBO FROZEN at step %d  θ* = %.4f",
                     frozen_at, optimizer.frozen_crossover_point)
            curve.append(snap(frozen_at))
            break

    if frozen_at == -1:
        if optimizer.apply_phase1_fallback():
            frozen_at = MAX_TRAINING_QUERIES
            log.info("  Phase-1 fallback applied  θ* = %.4f",
                     optimizer.frozen_crossover_point)
        else:
            log.warning("  CBO did NOT freeze within %d queries.", MAX_TRAINING_QUERIES)
            frozen_at = MAX_TRAINING_QUERIES
        curve.append(snap(MAX_TRAINING_QUERIES))

    return frozen_at, curve


# ── Frozen evaluation ─────────────────────────────────────────────────────────

def evaluate_phase(
    optimizer: ContextualBanditOptimizer,
    predictor: MiniBitmapPredictor,
    filters: List[FilterSpec],
    cache: Dict,
    n_queries: int,
) -> List[FilterResult]:
    log.info("  Evaluating frozen CBO (%d filters × %d queries) …",
             len(filters), n_queries)
    rows: List[FilterResult] = []

    for fi, fspec in enumerate(tqdm(filters, desc="  Frozen eval")):
        sel = fspec.actual_selectivity

        cbo_lats, cbo_recs, cbo_rews = [], [], []
        bm_lats,  bm_recs,  bm_rews  = [], [], []
        pf_lats,  pf_recs,  pf_rews  = [], [], []
        oracle_rews, cbo_opt          = [], []

        for qi in range(n_queries):
            est, _ = predictor.estimate_timed(fspec)
            strat  = optimizer.route(est)
            lat, rec = cache[strat][fi][qi]
            cbo_r = reward_fn(lat, rec)
            cbo_lats.append(lat)
            cbo_recs.append(rec)
            cbo_rews.append(cbo_r)

            bm_lat, bm_rec = cache["bitmap_prefilter"][fi][qi]
            bm_r = reward_fn(bm_lat, bm_rec)
            bm_lats.append(bm_lat); bm_recs.append(bm_rec); bm_rews.append(bm_r)

            pf_lat, pf_rec = cache["post_filter"][fi][qi]
            pf_r = reward_fn(pf_lat, pf_rec)
            pf_lats.append(pf_lat); pf_recs.append(pf_rec); pf_rews.append(pf_r)

            best_r = max(bm_r, pf_r)
            oracle_rews.append(best_r)
            best_strat = "bitmap_prefilter" if bm_r >= pf_r else "post_filter"
            cbo_opt.append(1.0 if strat == best_strat else 0.0)

        rows.append(FilterResult(
            filter_name=fspec.name,
            filter_type=classify_filter(fspec.name),
            selectivity_pct=round(sel * 100, 4),
            zone=zone_for(sel),
            cbo_lat_p50=round(pct(cbo_lats, 50), 4),
            cbo_lat_p95=round(pct(cbo_lats, 95), 4),
            cbo_recall=round(float(np.mean(cbo_recs)), 4),
            cbo_sla=round(sla(cbo_recs), 4),
            cbo_reward=round(float(np.mean(cbo_rews)), 4),
            oracle_reward=round(float(np.mean(oracle_rews)), 4),
            oracle_gap=round(float(np.mean(oracle_rews)) - float(np.mean(cbo_rews)), 4),
            cbo_optimal_rate=round(float(np.mean(cbo_opt)), 4),
            bm_lat_p50=round(pct(bm_lats, 50), 4),
            bm_lat_p95=round(pct(bm_lats, 95), 4),
            bm_recall=round(float(np.mean(bm_recs)), 4),
            bm_sla=round(sla(bm_recs), 4),
            bm_reward=round(float(np.mean(bm_rews)), 4),
            pf_lat_p50=round(pct(pf_lats, 50), 4),
            pf_lat_p95=round(pct(pf_lats, 95), 4),
            pf_recall=round(float(np.mean(pf_recs)), 4),
            pf_sla=round(sla(pf_recs), 4),
            pf_reward=round(float(np.mean(pf_rews)), 4),
        ))

    return rows


# ── Aggregation helpers ───────────────────────────────────────────────────────

def zone_summary(rows: List[FilterResult]) -> List[dict]:
    zones = defaultdict(list)
    for r in rows:
        zones[r.zone].append(r)
    out = []
    for label, _, _ in SELECTIVITY_ZONES:
        zr = zones.get(label, [])
        if not zr:
            continue
        out.append({
            "zone":           label,
            "n_filters":      len(zr),
            "cbo_sla":        round(float(np.mean([r.cbo_sla    for r in zr])), 4),
            "cbo_reward":     round(float(np.mean([r.cbo_reward  for r in zr])), 4),
            "oracle_gap":     round(float(np.mean([r.oracle_gap  for r in zr])), 4),
            "cbo_lat_p50":    round(float(np.mean([r.cbo_lat_p50 for r in zr])), 4),
            "cbo_lat_p95":    round(float(np.mean([r.cbo_lat_p95 for r in zr])), 4),
            "bm_sla":         round(float(np.mean([r.bm_sla      for r in zr])), 4),
            "bm_reward":      round(float(np.mean([r.bm_reward   for r in zr])), 4),
            "pf_sla":         round(float(np.mean([r.pf_sla      for r in zr])), 4),
            "pf_reward":      round(float(np.mean([r.pf_reward   for r in zr])), 4),
        })
    return out


def global_metrics(rows: List[FilterResult]) -> dict:
    return {
        "cbo_sla_rate":      round(float(np.mean([r.cbo_sla    for r in rows])), 4),
        "cbo_reward_mean":   round(float(np.mean([r.cbo_reward  for r in rows])), 4),
        "bm_reward_mean":    round(float(np.mean([r.bm_reward   for r in rows])), 4),
        "pf_reward_mean":    round(float(np.mean([r.pf_reward   for r in rows])), 4),
        "oracle_gap_mean":   round(float(np.mean([r.oracle_gap  for r in rows])), 4),
        "cbo_optimal_rate":  round(float(np.mean([r.cbo_optimal_rate for r in rows])), 4),
    }


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_per_filter(rows: List[FilterResult], path: Path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def save_learning_curve(curve: List[LcPoint], path: Path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(curve[0]).keys()))
        w.writeheader()
        for pt in curve:
            w.writerow(asdict(pt))


def print_phase_summary(
    n_docs: int,
    l_max: float,
    frozen_at: int,
    theta: Optional[float],
    gm: dict,
    zones: List[dict],
):
    print(f"""
{'='*72}
  Phase: {n_docs:,} docs  |  L_MAX={l_max:.1f}ms  |  frozen@{frozen_at}  |  θ*={theta}
{'='*72}
  Strategy         SLA      Avg Reward
  CBO              {gm['cbo_sla_rate']*100:5.1f}%    {gm['cbo_reward_mean']:.4f}
  Always Bitmap    {'n/a':>5}     {gm['bm_reward_mean']:.4f}
  Always Post      {'n/a':>5}     {gm['pf_reward_mean']:.4f}
  Oracle gap                  {gm['oracle_gap_mean']:.4f}
  CBO optimal rate            {gm['cbo_optimal_rate']*100:.1f}%

  Zone        N  CBO SLA  BM SLA  PF SLA  Orc Gap  CBO P50 lat
  {'-'*62}""")
    for z in zones:
        print(f"  {z['zone']:<10} {z['n_filters']:2d}  "
              f"{z['cbo_sla']*100:5.1f}%  {z['bm_sla']*100:5.1f}%  "
              f"{z['pf_sla']*100:5.1f}%  {z['oracle_gap']:.4f}   "
              f"{z['cbo_lat_p50']:7.1f}ms")
    print(f"{'='*72}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Incremental CBO benchmark")
    p.add_argument("--corpus-sizes", default=",".join(map(str, DEFAULT_CORPUS_SIZES)),
                   help="Comma-separated corpus sizes (default: 200000,335000,500000,671750)")
    p.add_argument("--n-queries", type=int, default=200,
                   help="Queries per filter per phase (default: 200)")
    p.add_argument("--n-warmup", type=int, default=5000,
                   help="N_WARMUP for Phase-2 resampling (default: 5000)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    corpus_sizes = [int(x) for x in args.corpus_sizes.split(",")]

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = config.RESULTS_DIR / f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)
    log.info("Corpus sizes: %s", corpus_sizes)
    log.info("N_WARMUP: %d  |  n_queries: %d  |  seed: %d",
             args.n_warmup, args.n_queries, args.seed)

    # ── Load base data once ────────────────────────────────────────────────────
    full_emb, full_meta = load_base(config.INDEX_DIR)
    available = full_emb.shape[0]
    corpus_sizes = [min(s, available) for s in corpus_sizes]

    # ── Override N_WARMUP ─────────────────────────────────────────────────────
    config.N_WARMUP = args.n_warmup

    # ── Phase loop ─────────────────────────────────────────────────────────────
    evolution   = []     # summary record per phase
    summary_rows = []    # for summary_table.csv

    for phase_idx, n_docs in enumerate(corpus_sizes, 1):
        log.info("")
        log.info("━" * 72)
        log.info("PHASE %d / %d  —  %d documents", phase_idx, len(corpus_sizes), n_docs)
        log.info("━" * 72)

        # Adaptive L_MAX for this corpus size
        l_max = config.compute_l_max(n_docs)
        config.CBO_L_MAX = l_max
        log.info("  Adaptive L_MAX = %.1f ms", l_max)

        # Build corpus
        emb, meta, faiss_idx, bmap_idx = build_phase_corpus(n_docs, full_emb, full_meta)

        # Strategies
        brute   = BruteForce(faiss_idx, meta)
        post_f  = PostFilter(faiss_idx, meta, expansion_factor=config.POST_FILTER_EXPANSION)
        bmap_f  = BitmapPreFilter(faiss_idx, bmap_idx)

        # Filters & queries
        predictor = MiniBitmapPredictor(meta,
                                        sample_ratio=config.PREDICTOR_SAMPLE_RATIO,
                                        seed=args.seed)
        filters = generate_filters(meta, config.SELECTIVITY_TARGETS)
        log.info("  Generated %d filters", len(filters))

        q_idx = random.sample(range(n_docs), args.n_queries)
        q_emb = emb[q_idx]

        # Cache
        cache = precompute(filters, q_emb, brute, post_f, bmap_f, args.n_queries)

        # Train
        optimizer = ContextualBanditOptimizer(n_corpus=n_docs, mode="epsilon_greedy")
        frozen_at, curve = train_phase(
            optimizer, predictor, filters, cache, args.n_queries, args.n_warmup
        )

        theta = optimizer.frozen_crossover_point

        # Theoretical L_norm at theta* (validates formula)
        theoretical_l_norm = None
        if theta is not None:
            # Estimate bitmap latency at theta* from Phase-1 data (approx):
            # bitmap latency scales as n_docs / 200k * baseline.
            # We track it as reported in per-filter results.
            pass

        # Evaluate
        rows = evaluate_phase(optimizer, predictor, filters, cache, args.n_queries)
        gm   = global_metrics(rows)
        zones = zone_summary(rows)

        # Empirical crossover: first sel where post_reward > bm_reward
        crossover_empirical = None
        sorted_rows = sorted(rows, key=lambda r: r.selectivity_pct)
        for i in range(len(sorted_rows) - 1):
            a, b = sorted_rows[i], sorted_rows[i + 1]
            if a.bm_reward > a.pf_reward and b.bm_reward <= b.pf_reward:
                margin_a = a.bm_reward - a.pf_reward
                margin_b = b.pf_reward - b.bm_reward
                # Linear interpolation
                cross = a.selectivity_pct + margin_a / (margin_a + margin_b) * (
                    b.selectivity_pct - a.selectivity_pct
                )
                crossover_empirical = round(cross, 2)
                break

        # Bitmap L_norm at theta* (actual)
        bm_l_norm_at_theta = None
        if theta is not None:
            # Find closest filter to theta*
            closest = min(rows, key=lambda r: abs(r.selectivity_pct - theta * 100))
            bm_l_norm_at_theta = round(
                max(0.0, 1.0 - closest.bm_lat_p50 / l_max), 4
            )

        phase_summary = {
            "phase":                  phase_idx,
            "n_docs":                 n_docs,
            "l_max":                  round(l_max, 2),
            "n_warmup":               args.n_warmup,
            "frozen_at_step":         frozen_at,
            "crossover_theta":        round(theta, 4) if theta else None,
            "crossover_empirical":    crossover_empirical,
            "bm_l_norm_at_theta":     bm_l_norm_at_theta,
            "cbo_sla_rate":           gm["cbo_sla_rate"],
            "cbo_reward_mean":        gm["cbo_reward_mean"],
            "bm_reward_mean":         gm["bm_reward_mean"],
            "pf_reward_mean":         gm["pf_reward_mean"],
            "oracle_gap_mean":        gm["oracle_gap_mean"],
            "cbo_optimal_rate":       gm["cbo_optimal_rate"],
            "zone_summaries":         zones,
        }
        evolution.append(phase_summary)
        summary_rows.append({
            "phase":        phase_idx,
            "n_docs":       n_docs,
            "l_max_ms":     round(l_max, 1),
            "n_warmup":     args.n_warmup,
            "frozen_step":  frozen_at,
            "theta_star":   round(theta, 4) if theta else "N/A",
            "theta_empirical": crossover_empirical if crossover_empirical else "N/A",
            "bm_l_norm_at_theta": bm_l_norm_at_theta if bm_l_norm_at_theta else "N/A",
            "cbo_reward":   gm["cbo_reward_mean"],
            "pf_reward":    gm["pf_reward_mean"],
            "bm_reward":    gm["bm_reward_mean"],
            "oracle_gap":   gm["oracle_gap_mean"],
            "cbo_vs_post":  round(gm["cbo_reward_mean"] - gm["pf_reward_mean"], 4),
            "cbo_sla":      gm["cbo_sla_rate"],
        })

        print_phase_summary(n_docs, l_max, frozen_at, theta, gm, zones)

        # ── Save phase files ───────────────────────────────────────────────────
        tag = f"phase_{phase_idx}_{n_docs}"
        with open(out_dir / f"{tag}_summary.json", "w") as fh:
            json.dump(phase_summary, fh, indent=2)

        save_per_filter(rows, out_dir / f"{tag}_per_filter.csv")
        save_learning_curve(curve, out_dir / f"{tag}_learning.csv")

        # Save Q-table snapshot for this phase
        with open(out_dir / f"{tag}_qtable.json", "w") as fh:
            json.dump(optimizer.get_q_snapshot(), fh, indent=2)

        log.info("  Phase %d saved → %s", phase_idx, out_dir)

    # ── Final outputs ──────────────────────────────────────────────────────────
    with open(out_dir / "crossover_evolution.json", "w") as fh:
        json.dump(evolution, fh, indent=2)

    with open(out_dir / "summary_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # ── Console summary table ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  INCREMENTAL CBO BENCHMARK — CROSS-SCALE SUMMARY")
    print(f"{'='*90}")
    print(f"  {'n_docs':>8}  {'L_MAX':>8}  {'θ*(learned)':>12}  {'θ*(empirical)':>14}"
          f"  {'BM L_norm@θ':>12}  {'CBO Rew':>8}  {'OrcGap':>8}  {'vs Post':>8}")
    print(f"  {'-'*88}")
    for r in summary_rows:
        print(f"  {r['n_docs']:>8,}  {r['l_max_ms']:>7.0f}ms"
              f"  {str(r['theta_star']):>12}  {str(r['theta_empirical']):>14}"
              f"  {str(r['bm_l_norm_at_theta']):>12}"
              f"  {r['cbo_reward']:>8.4f}  {r['oracle_gap']:>8.4f}"
              f"  {r['cbo_vs_post']:>+8.4f}")
    print(f"{'='*90}")
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

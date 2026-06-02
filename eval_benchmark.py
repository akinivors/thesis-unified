#!/usr/bin/env python3
"""
eval_benchmark.py
=================
Comprehensive CBO evaluation suite.

Fills the gaps left by main_benchmark.py and incremental_benchmark.py:

  1. Oracle gap       – per-query: reward of best available strategy vs CBO choice
  2. SLA compliance   – per selectivity zone (0-10%, 10-20%, 20-30%, 30-40%, 40-50%, 50-90%)
  3. Latency pct      – P50 / P95 / P99 per strategy × zone
  4. Learning curve   – Q-value snapshot + running SLA every 100 training steps
  5. Baselines        – "always bitmap" / "always post" for CBO value attribution
  6. Filter types     – per-type SLA breakdown (single / compound / range / in_set)
  7. 200 queries/filter (4× main_benchmark.py) for reliable percentile estimates
  8. Shadow-zone probe – detailed look at the 30-40% "no man's land" problem

Outputs (all written to results/eval_<timestamp>/):
  eval_summary.json        – all aggregate metrics in one place
  per_filter_results.csv   – filter-level breakdown (latency, recall, SLA, oracle gap)
  learning_curve.csv       – training step × Q-values × running SLA rate
  zone_summary.csv         – zone-level SLA / latency / oracle gap table
  console                  – human-readable summary at the end

Usage:
    python eval_benchmark.py
    python eval_benchmark.py --n-docs 100000 --n-queries 100 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
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

SELECTIVITY_ZONES = [
    ("0-10%",   0.00, 0.10),
    ("10-20%",  0.10, 0.20),
    ("20-30%",  0.20, 0.30),
    ("30-40%",  0.30, 0.40),   # "shadow zone" — CBO routes post, post recall issues
    ("40-50%",  0.40, 0.50),
    ("50-90%",  0.50, 0.90),
]

STRATEGY_NAMES = ("bitmap_prefilter", "post_filter")

MAX_TRAINING_QUERIES = 15_000   # hard cap; bandit should freeze well before this


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FilterResultRow:
    """Per-filter aggregate metrics for the CSV."""
    filter_name: str
    filter_type: str        # single / compound / range / in_set
    selectivity_pct: float
    zone: str

    # CBO (frozen evaluation)
    cbo_latency_p50: float
    cbo_latency_p95: float
    cbo_latency_p99: float
    cbo_recall_mean: float
    cbo_sla_rate: float       # fraction of queries meeting R_TARGET
    cbo_reward_mean: float

    # Oracle (best available arm per query)
    oracle_reward_mean: float
    oracle_gap: float          # oracle_reward - cbo_reward (≥ 0)
    cbo_optimal_rate: float    # fraction of queries CBO chose the oracle arm

    # Always-bitmap baseline
    bitmap_latency_p50: float
    bitmap_latency_p95: float
    bitmap_recall_mean: float
    bitmap_sla_rate: float
    bitmap_reward_mean: float

    # Always-post baseline
    post_latency_p50: float
    post_latency_p95: float
    post_recall_mean: float
    post_sla_rate_mean: float
    post_reward_mean: float


@dataclass
class LearningCurvePoint:
    """Snapshot taken every LEARNING_SNAPSHOT_INTERVAL training queries."""
    training_step: int
    is_frozen: bool
    crossover_estimate: Optional[float]
    sla_rate_last_100: float           # running SLA rate over last 100 training queries
    reward_mean_last_100: float
    # Q-values at a few representative selectivities for trend analysis
    q_bitmap_at_020: float
    q_post_at_020: float
    q_bitmap_at_030: float
    q_post_at_030: float
    q_bitmap_at_040: float
    q_post_at_040: float


# ── Helper functions ───────────────────────────────────────────────────────────

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


def sla_rate(recalls: list[float], target: float = config.CBO_R_TARGET) -> float:
    if not recalls:
        return 0.0
    return sum(1 for r in recalls if r >= target) / len(recalls)


def reward_fn(lat: float, rec: float) -> float:
    """Inline SoftCliff reward — mirrors cbo.reward.SoftCliffReward."""
    margin = config.CBO_RECALL_MARGIN
    if rec < config.CBO_R_TARGET - margin:
        return 0.0
    l_norm = max(0.0, 1.0 - lat / config.CBO_L_MAX)
    if rec >= config.CBO_R_TARGET:
        return l_norm
    ratio = rec / config.CBO_R_TARGET
    return l_norm * (ratio ** config.CBO_BETA)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


# ── Build corpus ──────────────────────────────────────────────────────────────

def build_corpus(n_docs: int) -> Tuple[np.ndarray, list, FAISSIndex, BitmapIndex]:
    log.info("Loading index from %s …", config.INDEX_DIR)
    full_faiss = FAISSIndex.load(config.INDEX_DIR)
    with open(config.INDEX_DIR / "all_metadatas.pkl", "rb") as fh:
        full_meta = pickle.load(fh)

    n_docs = min(n_docs, full_faiss.hnsw_index.ntotal, len(full_meta))
    log.info("Using %d documents.", n_docs)

    embeddings = full_faiss._embeddings[:n_docs]
    metadata   = full_meta[:n_docs]

    log.info("Building FAISS HNSW index (%d docs) …", n_docs)
    faiss_idx = FAISSIndex(dim=embeddings.shape[1])
    faiss_idx.build(embeddings, [str(i) for i in range(n_docs)])

    log.info("Building BitmapIndex …")
    bitmap_idx = BitmapIndex()
    for i, doc in enumerate(metadata):
        bitmap_idx.add_document(i, str(i), doc)

    return embeddings, metadata, faiss_idx, bitmap_idx


# ── Pre-compute cache ─────────────────────────────────────────────────────────

def precompute_cache(
    filters: List[FilterSpec],
    query_embeddings: np.ndarray,
    brute_force: BruteForce,
    post_filter: PostFilter,
    bitmap_prefilter: BitmapPreFilter,
    n_queries: int,
) -> Dict:
    """
    For each (filter, query) pre-compute ground truth and both strategy results.

    cache["bitmap_prefilter"][f_idx][q_idx] = (latency_ms, recall)
    cache["post_filter"][f_idx][q_idx]      = (latency_ms, recall)
    """
    cache = {
        "bitmap_prefilter": [[None] * n_queries for _ in range(len(filters))],
        "post_filter":      [[None] * n_queries for _ in range(len(filters))],
    }

    for f_idx, fspec in enumerate(tqdm(filters, desc="Pre-computing cache")):
        sel = fspec.actual_selectivity
        for q_idx in range(n_queries):
            q_emb = query_embeddings[q_idx].reshape(1, -1)

            bf_res        = brute_force.search(q_emb, config.CBO_TOP_K, fspec)
            true_ids      = set(bf_res.ids)
            n_true        = len(true_ids) if true_ids else 1

            pf_res = post_filter.search(q_emb, config.CBO_TOP_K, fspec, selectivity=sel)
            pf_recall = len(set(pf_res.ids) & true_ids) / n_true
            cache["post_filter"][f_idx][q_idx] = (pf_res.total_time_ms, pf_recall)

            bp_res = bitmap_prefilter.search(q_emb, config.CBO_TOP_K, fspec, selectivity=sel)
            bp_recall = len(set(bp_res.ids) & true_ids) / n_true
            cache["bitmap_prefilter"][f_idx][q_idx] = (bp_res.total_time_ms, bp_recall)

    return cache


# ── CBO Training with learning-curve tracking ─────────────────────────────────

LEARNING_SNAPSHOT_INTERVAL = 100   # snapshot every N training queries

def train_cbo(
    optimizer: ContextualBanditOptimizer,
    predictor: MiniBitmapPredictor,
    filters: List[FilterSpec],
    cache: Dict,
    n_queries: int,
) -> Tuple[int, List[LearningCurvePoint]]:
    """
    Train the CBO until it freezes (or MAX_TRAINING_QUERIES is reached).
    Returns (frozen_at_query, learning_curve).
    """
    log.info("Starting CBO training (max %d queries, N_WARMUP=%d) …",
             MAX_TRAINING_QUERIES, optimizer.n_warmup)

    curve: List[LearningCurvePoint] = []
    frozen_at = -1
    window_rewards: List[float] = []
    window_sla: List[float] = []

    def snapshot(step: int) -> LearningCurvePoint:
        sla = sla_rate(window_sla[-100:]) if window_sla else 0.0
        rew = float(np.mean(window_rewards[-100:])) if window_rewards else 0.0
        cross = optimizer.get_crossover_estimate()

        def q_pair(sel: float):
            qv = optimizer.qtable.get_q_values(sel)
            return qv.get("bitmap_prefilter", 1.0), qv.get("post_filter", 1.0)

        bm20, po20 = q_pair(0.20)
        bm30, po30 = q_pair(0.30)
        bm40, po40 = q_pair(0.40)

        return LearningCurvePoint(
            training_step=step,
            is_frozen=optimizer.is_frozen,
            crossover_estimate=cross,
            sla_rate_last_100=round(sla, 4),
            reward_mean_last_100=round(rew, 4),
            q_bitmap_at_020=round(bm20, 4),
            q_post_at_020=round(po20, 4),
            q_bitmap_at_030=round(bm30, 4),
            q_post_at_030=round(po30, 4),
            q_bitmap_at_040=round(bm40, 4),
            q_post_at_040=round(po40, 4),
        )

    for step in range(MAX_TRAINING_QUERIES):
        fspec = random.choice(filters)
        q_idx = random.randrange(n_queries)
        f_idx = filters.index(fspec)

        est_sel, _ = predictor.estimate_timed(fspec)
        strategy   = optimizer.route(est_sel)

        latency_ms, recall = cache[strategy][f_idx][q_idx]
        reward = optimizer.feedback(est_sel, strategy, latency_ms, recall)

        window_rewards.append(reward)
        window_sla.append(float(recall >= config.CBO_R_TARGET))

        if (step + 1) % LEARNING_SNAPSHOT_INTERVAL == 0:
            curve.append(snapshot(step + 1))
            log.info(
                "  [step %5d] frozen=%s  crossover=%-6s  sla_last100=%.1f%%  "
                "q_bitmap@0.30=%.3f  q_post@0.30=%.3f",
                step + 1,
                optimizer.is_frozen,
                f"{optimizer.frozen_crossover_point:.3f}" if optimizer.frozen_crossover_point else "None",
                window_sla[-100:].count(1.0) / min(100, len(window_sla)) * 100,
                optimizer.qtable.get_q_values(0.30).get("bitmap_prefilter", 1.0),
                optimizer.qtable.get_q_values(0.30).get("post_filter", 1.0),
            )

        if optimizer.is_frozen and frozen_at == -1:
            frozen_at = step + 1
            log.info("CBO FROZEN at training step %d!  θ* = %.4f",
                     frozen_at, optimizer.frozen_crossover_point)
            # Take a final snapshot immediately on freeze
            curve.append(snapshot(frozen_at))
            break

    if frozen_at == -1:
        # Fix 2 (non-adjacent gap-bridging) runs inside the training loop via
        # feedback(); reaching here means it also did not trigger.
        # Attempt the Phase 1 fallback (Fix 3) as a final resort.
        if optimizer.apply_phase1_fallback():
            frozen_at = MAX_TRAINING_QUERIES   # mark step at which fallback applied
            log.info(
                "Phase 2 gap-bridge did not fire — applied Phase 1 fallback θ* = %.4f",
                optimizer.frozen_crossover_point,
            )
        else:
            log.warning("CBO did NOT freeze within %d queries.", MAX_TRAINING_QUERIES)
        frozen_at = MAX_TRAINING_QUERIES
        curve.append(snapshot(frozen_at))

    return frozen_at, curve


# ── Frozen evaluation ─────────────────────────────────────────────────────────

def evaluate_frozen(
    optimizer: ContextualBanditOptimizer,
    predictor: MiniBitmapPredictor,
    filters: List[FilterSpec],
    cache: Dict,
    n_queries: int,
) -> List[FilterResultRow]:
    """
    For each filter, run all n_queries through the frozen CBO and collect
    per-query strategy, latency, recall, and reward.  Also compute the oracle.
    Returns one FilterResultRow per filter.
    """
    log.info("Running frozen evaluation (%d filters × %d queries) …",
             len(filters), n_queries)
    rows: List[FilterResultRow] = []

    for f_idx, fspec in enumerate(tqdm(filters, desc="Frozen eval")):
        sel = fspec.actual_selectivity

        cbo_lats, cbo_recs, cbo_rewards      = [], [], []
        oracle_rewards                        = []
        cbo_chose_oracle                      = []

        for q_idx in range(n_queries):
            est_sel, _ = predictor.estimate_timed(fspec)
            strategy   = optimizer.route(est_sel)

            lat, rec = cache[strategy][f_idx][q_idx]
            rew = reward_fn(lat, rec)

            cbo_lats.append(lat)
            cbo_recs.append(rec)
            cbo_rewards.append(rew)

            # Oracle: best reward over both arms using actual selectivity
            bitmap_lat, bitmap_rec = cache["bitmap_prefilter"][f_idx][q_idx]
            post_lat,   post_rec   = cache["post_filter"][f_idx][q_idx]
            bitmap_rew = reward_fn(bitmap_lat, bitmap_rec)
            post_rew   = reward_fn(post_lat,   post_rec)

            oracle_rew = max(bitmap_rew, post_rew)
            oracle_rewards.append(oracle_rew)

            oracle_arm = ("bitmap_prefilter" if bitmap_rew >= post_rew else "post_filter")
            cbo_chose_oracle.append(float(strategy == oracle_arm))

        # Always-bitmap metrics
        bm_lats = [cache["bitmap_prefilter"][f_idx][q][0] for q in range(n_queries)]
        bm_recs = [cache["bitmap_prefilter"][f_idx][q][1] for q in range(n_queries)]
        bm_rews = [reward_fn(l, r) for l, r in zip(bm_lats, bm_recs)]

        # Always-post metrics
        pf_lats = [cache["post_filter"][f_idx][q][0] for q in range(n_queries)]
        pf_recs = [cache["post_filter"][f_idx][q][1] for q in range(n_queries)]
        pf_rews = [reward_fn(l, r) for l, r in zip(pf_lats, pf_recs)]

        cbo_reward_mean    = float(np.mean(cbo_rewards))
        oracle_reward_mean = float(np.mean(oracle_rewards))
        oracle_gap         = oracle_reward_mean - cbo_reward_mean  # always ≥ 0

        rows.append(FilterResultRow(
            filter_name   = fspec.name,
            filter_type   = classify_filter(fspec.name),
            selectivity_pct = round(sel * 100, 2),
            zone          = zone_for(sel),

            cbo_latency_p50  = round(percentile(cbo_lats, 50), 3),
            cbo_latency_p95  = round(percentile(cbo_lats, 95), 3),
            cbo_latency_p99  = round(percentile(cbo_lats, 99), 3),
            cbo_recall_mean  = round(float(np.mean(cbo_recs)), 4),
            cbo_sla_rate     = round(sla_rate(cbo_recs), 4),
            cbo_reward_mean  = round(cbo_reward_mean, 4),

            oracle_reward_mean = round(oracle_reward_mean, 4),
            oracle_gap         = round(oracle_gap, 4),
            cbo_optimal_rate   = round(float(np.mean(cbo_chose_oracle)), 4),

            bitmap_latency_p50  = round(percentile(bm_lats, 50), 3),
            bitmap_latency_p95  = round(percentile(bm_lats, 95), 3),
            bitmap_recall_mean  = round(float(np.mean(bm_recs)), 4),
            bitmap_sla_rate     = round(sla_rate(bm_recs), 4),
            bitmap_reward_mean  = round(float(np.mean(bm_rews)), 4),

            post_latency_p50   = round(percentile(pf_lats, 50), 3),
            post_latency_p95   = round(percentile(pf_lats, 95), 3),
            post_recall_mean   = round(float(np.mean(pf_recs)), 4),
            post_sla_rate_mean = round(sla_rate(pf_recs), 4),
            post_reward_mean   = round(float(np.mean(pf_rews)), 4),
        ))

    return rows


# ── Zone aggregation ───────────────────────────────────────────────────────────

def aggregate_by_zone(rows: List[FilterResultRow]) -> List[Dict]:
    """Roll up FilterResultRows into per-zone summary dicts."""
    buckets: Dict[str, List[FilterResultRow]] = defaultdict(list)
    for r in rows:
        buckets[r.zone].append(r)

    zone_summaries = []
    for label, lo, hi in SELECTIVITY_ZONES:
        zone_rows = buckets[label]
        if not zone_rows:
            continue

        def mean(vals):
            return round(float(np.mean(vals)), 4)

        zone_summaries.append({
            "zone":                label,
            "n_filters":           len(zone_rows),
            "sel_range":           f"{lo*100:.0f}%-{hi*100:.0f}%",

            # CBO
            "cbo_sla_rate":        mean([r.cbo_sla_rate      for r in zone_rows]),
            "cbo_reward_mean":     mean([r.cbo_reward_mean    for r in zone_rows]),
            "cbo_lat_p50":         mean([r.cbo_latency_p50    for r in zone_rows]),
            "cbo_lat_p95":         mean([r.cbo_latency_p95    for r in zone_rows]),
            "cbo_lat_p99":         mean([r.cbo_latency_p99    for r in zone_rows]),
            "cbo_recall_mean":     mean([r.cbo_recall_mean    for r in zone_rows]),

            # Oracle gap
            "oracle_reward_mean":  mean([r.oracle_reward_mean for r in zone_rows]),
            "oracle_gap":          mean([r.oracle_gap         for r in zone_rows]),
            "cbo_optimal_rate":    mean([r.cbo_optimal_rate   for r in zone_rows]),

            # Always-bitmap baseline
            "bitmap_sla_rate":     mean([r.bitmap_sla_rate    for r in zone_rows]),
            "bitmap_reward_mean":  mean([r.bitmap_reward_mean for r in zone_rows]),
            "bitmap_lat_p50":      mean([r.bitmap_latency_p50 for r in zone_rows]),
            "bitmap_lat_p95":      mean([r.bitmap_latency_p95 for r in zone_rows]),

            # Always-post baseline
            "post_sla_rate":       mean([r.post_sla_rate_mean for r in zone_rows]),
            "post_reward_mean":    mean([r.post_reward_mean   for r in zone_rows]),
            "post_lat_p50":        mean([r.post_latency_p50   for r in zone_rows]),
            "post_lat_p95":        mean([r.post_latency_p95   for r in zone_rows]),
        })

    return zone_summaries


# ── Filter-type breakdown ─────────────────────────────────────────────────────

def aggregate_by_filter_type(rows: List[FilterResultRow]) -> List[Dict]:
    buckets: Dict[str, List[FilterResultRow]] = defaultdict(list)
    for r in rows:
        buckets[r.filter_type].append(r)

    result = []
    for ftype, type_rows in sorted(buckets.items()):
        def mean(vals):
            return round(float(np.mean(vals)), 4)

        result.append({
            "filter_type":          ftype,
            "n_filters":            len(type_rows),
            "cbo_sla_rate":         mean([r.cbo_sla_rate      for r in type_rows]),
            "post_sla_rate":        mean([r.post_sla_rate_mean for r in type_rows]),
            "bitmap_sla_rate":      mean([r.bitmap_sla_rate    for r in type_rows]),
            "oracle_gap":           mean([r.oracle_gap         for r in type_rows]),
            "cbo_optimal_rate":     mean([r.cbo_optimal_rate   for r in type_rows]),
            "cbo_recall_mean":      mean([r.cbo_recall_mean    for r in type_rows]),
            "post_recall_mean":     mean([r.post_recall_mean   for r in type_rows]),
        })

    return result


# ── Pretty-print summary ───────────────────────────────────────────────────────

def print_summary(
    frozen_at: int,
    crossover: Optional[float],
    zone_summaries: List[Dict],
    filter_type_summaries: List[Dict],
    rows: List[FilterResultRow],
):
    sep = "=" * 80

    print(f"\n{sep}")
    print("  BANDIT-CBO COMPREHENSIVE EVALUATION SUMMARY")
    print(sep)
    print(f"  Frozen at training step : {frozen_at}")
    print(f"  Learned θ*             : {crossover:.4f}" if crossover else "  Learned θ*             : NOT FROZEN")

    # Global aggregates
    all_cbo_sla   = float(np.mean([r.cbo_sla_rate     for r in rows]))
    all_bm_sla    = float(np.mean([r.bitmap_sla_rate  for r in rows]))
    all_pf_sla    = float(np.mean([r.post_sla_rate_mean for r in rows]))
    all_cbo_rew   = float(np.mean([r.cbo_reward_mean  for r in rows]))
    all_bm_rew    = float(np.mean([r.bitmap_reward_mean for r in rows]))
    all_pf_rew    = float(np.mean([r.post_reward_mean for r in rows]))
    all_oracle_gap = float(np.mean([r.oracle_gap      for r in rows]))
    all_opt_rate   = float(np.mean([r.cbo_optimal_rate for r in rows]))

    print(f"\n  {'Strategy':<20} {'SLA Rate':>10} {'Avg Reward':>12}")
    print(f"  {'-'*44}")
    print(f"  {'CBO (frozen)':<20} {all_cbo_sla*100:>9.1f}%  {all_cbo_rew:>12.4f}")
    print(f"  {'Always Bitmap':<20} {all_bm_sla*100:>9.1f}%  {all_bm_rew:>12.4f}")
    print(f"  {'Always Post':<20} {all_pf_sla*100:>9.1f}%  {all_pf_rew:>12.4f}")
    print(f"\n  Oracle gap  (reward left on table) : {all_oracle_gap:.4f}")
    print(f"  CBO chose optimal arm              : {all_opt_rate*100:.1f}% of queries")

    print(f"\n{sep}")
    print("  PER-ZONE BREAKDOWN")
    print(sep)
    hdr = f"  {'Zone':<10} {'N':>3} {'CBO SLA':>8} {'BM SLA':>8} {'PF SLA':>8} " \
          f"{'Orc Gap':>8} {'Opt%':>6} {'CBO P95':>9}"
    print(hdr)
    print(f"  {'-'*76}")
    for z in zone_summaries:
        print(
            f"  {z['zone']:<10} {z['n_filters']:>3} "
            f"{z['cbo_sla_rate']*100:>7.1f}% "
            f"{z['bitmap_sla_rate']*100:>7.1f}% "
            f"{z['post_sla_rate']*100:>7.1f}% "
            f"{z['oracle_gap']:>8.4f} "
            f"{z['cbo_optimal_rate']*100:>5.1f}% "
            f"{z['cbo_lat_p95']:>8.1f}ms"
        )

    print(f"\n{sep}")
    print("  BY FILTER TYPE")
    print(sep)
    hdr2 = f"  {'Type':<12} {'N':>4} {'CBO SLA':>9} {'PF SLA':>9} {'BM SLA':>9} {'Orc Gap':>9}"
    print(hdr2)
    print(f"  {'-'*56}")
    for ft in filter_type_summaries:
        print(
            f"  {ft['filter_type']:<12} {ft['n_filters']:>4} "
            f"{ft['cbo_sla_rate']*100:>8.1f}% "
            f"{ft['post_sla_rate']*100:>8.1f}% "
            f"{ft['bitmap_sla_rate']*100:>8.1f}% "
            f"{ft['oracle_gap']:>9.4f}"
        )

    # Shadow zone spotlight
    shadow = next((z for z in zone_summaries if z["zone"] == "30-40%"), None)
    if shadow:
        print(f"\n{sep}")
        print("  SHADOW-ZONE SPOTLIGHT  (30-40% selectivity — known problematic region)")
        print(sep)
        print(f"  CBO routes to post-filter in this zone (θ* = {crossover:.3f})" if crossover else "")
        print(f"  PostFilter SLA rate  : {shadow['post_sla_rate']*100:.1f}%")
        print(f"  BitmapBrute SLA rate : {shadow['bitmap_sla_rate']*100:.1f}%")
        print(f"  CBO SLA rate         : {shadow['cbo_sla_rate']*100:.1f}%")
        print(f"  Oracle gap           : {shadow['oracle_gap']:.4f}")
        print(f"  PostFilter P95 lat   : {shadow['post_lat_p95']:.1f}ms")
        print(f"  BitmapBrute P95 lat  : {shadow['bitmap_lat_p95']:.1f}ms")
        if shadow["post_sla_rate"] < shadow["bitmap_sla_rate"] - 0.05:
            print("  ⚠  Shadow zone confirmed: PostFilter recall lags BitmapBrute by "
                  f"{(shadow['bitmap_sla_rate'] - shadow['post_sla_rate'])*100:.1f}pp")
            print("     Consider lowering θ* or adding a mid-zone guard.")

    print(f"\n{sep}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive CBO evaluation benchmark.")
    p.add_argument("--n-docs",    type=int, default=200_000)
    p.add_argument("--n-queries", type=int, default=200,
                   help="Query samples per filter (default: 200; main_benchmark uses 50)")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = config.RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ── Build corpus & strategies ─────────────────────────────────────────────
    embeddings, metadata, faiss_idx, bitmap_idx = build_corpus(args.n_docs)
    n_docs = faiss_idx.hnsw_index.ntotal

    brute_force      = BruteForce(faiss_idx, metadata)
    post_filter      = PostFilter(faiss_idx, metadata, expansion_factor=config.POST_FILTER_EXPANSION)
    bitmap_prefilter = BitmapPreFilter(faiss_idx, bitmap_idx)

    predictor = MiniBitmapPredictor(
        metadata, sample_ratio=config.PREDICTOR_SAMPLE_RATIO, seed=args.seed
    )

    # ── Filters & queries ─────────────────────────────────────────────────────
    filters = generate_filters(metadata, config.SELECTIVITY_TARGETS)
    log.info("Generated %d filters.", len(filters))

    query_indices   = random.sample(range(n_docs), args.n_queries)
    query_embeddings = embeddings[query_indices]

    # ── Pre-compute cache ─────────────────────────────────────────────────────
    cache = precompute_cache(
        filters, query_embeddings,
        brute_force, post_filter, bitmap_prefilter,
        n_queries=args.n_queries,
    )

    # ── Adaptive L_MAX (Fix 1) ────────────────────────────────────────────────
    # Scale L_MAX linearly with corpus size so BitmapBrute remains competitive
    # up to the same relative selectivity region regardless of corpus scale.
    config.CBO_L_MAX = config.compute_l_max(n_docs)
    log.info("Adaptive L_MAX = %.1f ms  (n_docs=%d, base=%.1f ms @ %d docs)",
             config.CBO_L_MAX, n_docs, config.CBO_L_MAX_BASE, config.CBO_L_MAX_REF_DOCS)

    # ── CBO Training ──────────────────────────────────────────────────────────
    optimizer = ContextualBanditOptimizer(n_corpus=n_docs, mode="epsilon_greedy")
    frozen_at, learning_curve = train_cbo(
        optimizer, predictor, filters, cache, n_queries=args.n_queries
    )

    crossover = optimizer.get_crossover_estimate()
    qtable_snapshot = optimizer.get_q_snapshot()

    # ── Frozen Evaluation ─────────────────────────────────────────────────────
    filter_rows = evaluate_frozen(
        optimizer, predictor, filters, cache, n_queries=args.n_queries
    )

    # ── Aggregations ──────────────────────────────────────────────────────────
    zone_summaries       = aggregate_by_zone(filter_rows)
    filter_type_summaries = aggregate_by_filter_type(filter_rows)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(frozen_at, crossover, zone_summaries, filter_type_summaries, filter_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    # 1. per_filter_results.csv
    csv_path = out_dir / "per_filter_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(filter_rows[0]).keys()))
        writer.writeheader()
        for r in sorted(filter_rows, key=lambda x: x.selectivity_pct):
            writer.writerow(asdict(r))
    log.info("Saved per_filter_results.csv  (%d rows)", len(filter_rows))

    # 2. learning_curve.csv
    lc_path = out_dir / "learning_curve.csv"
    with open(lc_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(learning_curve[0]).keys()))
        writer.writeheader()
        for pt in learning_curve:
            writer.writerow(asdict(pt))
    log.info("Saved learning_curve.csv  (%d snapshots)", len(learning_curve))

    # 3. zone_summary.csv
    zone_path = out_dir / "zone_summary.csv"
    with open(zone_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(zone_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(zone_summaries)
    log.info("Saved zone_summary.csv")

    # 4. eval_summary.json
    summary = {
        "run_config": {
            "n_docs":      n_docs,
            "n_queries":   args.n_queries,
            "n_filters":   len(filters),
            "seed":        args.seed,
            "r_target":    config.CBO_R_TARGET,
            "l_max":       config.CBO_L_MAX,
            "n_warmup":    config.N_WARMUP,
        },
        "cbo_training": {
            "frozen_at_step":    frozen_at,
            "crossover_theta":   crossover,
            "is_frozen":         optimizer.is_frozen,
        },
        "global_metrics": {
            "cbo_sla_rate":      round(float(np.mean([r.cbo_sla_rate      for r in filter_rows])), 4),
            "bitmap_sla_rate":   round(float(np.mean([r.bitmap_sla_rate   for r in filter_rows])), 4),
            "post_sla_rate":     round(float(np.mean([r.post_sla_rate_mean for r in filter_rows])), 4),
            "cbo_reward_mean":   round(float(np.mean([r.cbo_reward_mean   for r in filter_rows])), 4),
            "bitmap_reward_mean":round(float(np.mean([r.bitmap_reward_mean for r in filter_rows])), 4),
            "post_reward_mean":  round(float(np.mean([r.post_reward_mean  for r in filter_rows])), 4),
            "oracle_gap_mean":   round(float(np.mean([r.oracle_gap        for r in filter_rows])), 4),
            "cbo_optimal_rate":  round(float(np.mean([r.cbo_optimal_rate  for r in filter_rows])), 4),
        },
        "zone_summaries":        zone_summaries,
        "filter_type_summaries": filter_type_summaries,
        "qtable_snapshot":       qtable_snapshot,
    }
    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Saved eval_summary.json")

    log.info("All outputs in %s", out_dir)


if __name__ == "__main__":
    main()

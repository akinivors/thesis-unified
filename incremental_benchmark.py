"""
incremental_benchmark.py
========================
Tests crossover adaptation across different corpus sizes (100k, 200k, 300k, 388k)
using the "Learn & Freeze" CBO architecture with MiniBitmapPredictor.
Precomputes all baselines (including IDSelector) for clean comparative analysis.
"""

import json
import csv
import random
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pickle
from tqdm import tqdm

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec, generate_filters
from strategies import BitmapPreFilter, BitmapHNSWPreFilter, PostFilter, BruteForce
from selectivity_predictor import MiniBitmapPredictor

from cbo import ContextualBanditOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_QUERIES = 50
RANDOM_SEED = 99
CORPUS_SIZES = [100_000, 200_000, 300_000, 388_000]
MAX_QUERIES_BEFORE_FREEZE = 5000

@dataclass
class QueryDetail:
    phase: str  # "learning" or "frozen"
    n_docs: int
    query_idx: int
    filter_idx: int
    actual_selectivity: float
    estimated_selectivity: float
    strategy_chosen: str
    latency_ms: float
    recall: float
    reward: float
    predict_overhead_us: float
    route_overhead_us: float


def build_corpus(n_docs: int) -> Tuple[np.ndarray, list, FAISSIndex, BitmapIndex]:
    logger.info("Loading full embeddings and metadata from %s …", config.INDEX_DIR)
    
    full_faiss_idx = FAISSIndex.load(config.INDEX_DIR)
    total_available = full_faiss_idx.hnsw_index.ntotal
    full_embeddings = full_faiss_idx._embeddings
    
    with open(config.INDEX_DIR / "all_metadatas.pkl", "rb") as fh:
        full_metadatas = pickle.load(fh)
        
    n_docs = min(n_docs, total_available)
    embeddings = full_embeddings[:n_docs]
    metadata = full_metadatas[:n_docs]
    
    logger.info("Building FAISS HNSW Index for %d docs...", n_docs)
    faiss_idx = FAISSIndex(dim=embeddings.shape[1])
    faiss_idx.build(embeddings, [str(i) for i in range(n_docs)])
    
    logger.info("Building Bitmap Index...")
    bitmap_idx = BitmapIndex()
    for i, doc in enumerate(metadata):
        bitmap_idx.add_document(i, str(i), doc)
        
    return embeddings, metadata, faiss_idx, bitmap_idx

def main():
    out_dir = config.RESULTS_DIR / f"incremental_freeze_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    crossover_evolution = []
    all_query_details = []

    for phase_idx, n_docs in enumerate(CORPUS_SIZES, 1):
        logger.info(f"\n{'='*40}\nPhase {phase_idx}: {n_docs} documents\n{'='*40}")
        embeddings, metadata, faiss_idx, bitmap_idx = build_corpus(n_docs)
        
        logger.info("Building MiniBitmapPredictor (Sample Ratio: %s)...", config.PREDICTOR_SAMPLE_RATIO)
        predictor = MiniBitmapPredictor(metadata, sample_ratio=config.PREDICTOR_SAMPLE_RATIO, seed=RANDOM_SEED)
        
        logger.info("Generating filters...")
        filters = generate_filters(metadata, config.SELECTIVITY_TARGETS)
        
        query_indices = random.sample(range(n_docs), NUM_QUERIES)
        query_embeddings = embeddings[query_indices]
        
        # ── Pre-computing Cache ────────────────────────────────────────────────
        logger.info("Pre-computing Ground Truth & Baselines for %d docs...", n_docs)
        cache = {
            "post_filter": [[None]*NUM_QUERIES for _ in range(len(filters))],
            "bitmap_prefilter": [[None]*NUM_QUERIES for _ in range(len(filters))],
            "bitmap_hnsw_prefilter": [[None]*NUM_QUERIES for _ in range(len(filters))],
        }
        
        brute_force = BruteForce(faiss_idx, metadata)
        post_filter = PostFilter(faiss_idx, metadata, expansion_factor=config.POST_FILTER_EXPANSION)
        bitmap_prefilter = BitmapPreFilter(faiss_idx, bitmap_idx)
        bitmap_hnsw_prefilter = BitmapHNSWPreFilter(faiss_idx, bitmap_idx)
        
        for f_idx, fspec in enumerate(tqdm(filters, desc="Pre-computing")):
            for q_idx in range(NUM_QUERIES):
                q_emb = query_embeddings[q_idx].reshape(1, -1)
                
                bf_result = brute_force.search(q_emb, config.CBO_TOP_K, fspec)
                true_neighbors = bf_result.ids
                
                pf_result = post_filter.search(q_emb, config.CBO_TOP_K, fspec)
                pf_recall = len(set(pf_result.ids) & set(true_neighbors)) / len(true_neighbors) if true_neighbors else 1.0
                cache["post_filter"][f_idx][q_idx] = (pf_result.total_time_ms, pf_recall)
                
                bp_result = bitmap_prefilter.search(q_emb, config.CBO_TOP_K, fspec)
                bp_recall = len(set(bp_result.ids) & set(true_neighbors)) / len(true_neighbors) if true_neighbors else 1.0
                cache["bitmap_prefilter"][f_idx][q_idx] = (bp_result.total_time_ms, bp_recall)

                bh_result = bitmap_hnsw_prefilter.search(q_emb, config.CBO_TOP_K, fspec)
                bh_recall = len(set(bh_result.ids) & set(true_neighbors)) / len(true_neighbors) if true_neighbors else 1.0
                cache["bitmap_hnsw_prefilter"][f_idx][q_idx] = (bh_result.total_time_ms, bh_recall)
        
        # ── Bandit CBO Training (Learn & Freeze) ──────────────────────────────
        logger.info("Running Bandit CBO (Learning Mode)...")
        optimizer = ContextualBanditOptimizer(n_corpus=n_docs, mode="epsilon_greedy")
        
        queries_executed = 0
        frozen_at_query = -1
        
        while not optimizer.is_frozen and queries_executed < MAX_QUERIES_BEFORE_FREEZE:
            fspec = random.choice(filters)
            q_idx = random.randrange(NUM_QUERIES)
            f_idx = filters.index(fspec)
            
            est_sel, predict_time_ms = predictor.estimate_timed(fspec)
            predict_overhead_us = predict_time_ms * 1000.0
            
            t0 = time.perf_counter()
            strategy_name = optimizer.route(est_sel)
            route_overhead_us = (time.perf_counter() - t0) * 1_000_000.0
            
            latency_ms, recall = cache[strategy_name][f_idx][q_idx]
            reward = optimizer.feedback(est_sel, strategy_name, latency_ms, recall)
            
            all_query_details.append(QueryDetail(
                phase="learning",
                n_docs=n_docs,
                query_idx=q_idx,
                filter_idx=f_idx,
                actual_selectivity=fspec.actual_selectivity,
                estimated_selectivity=est_sel,
                strategy_chosen=strategy_name,
                latency_ms=latency_ms,
                recall=recall,
                reward=reward,
                predict_overhead_us=predict_overhead_us,
                route_overhead_us=route_overhead_us
            ))
            
            queries_executed += 1
            if optimizer.is_frozen:
                frozen_at_query = queries_executed
                logger.info("CBO FROZEN at query %d! Stable crossover point: %.4f", frozen_at_query, optimizer.frozen_crossover_point)
                
        if not optimizer.is_frozen:
            logger.warning("CBO did not freeze within %d queries! Taking snapshot crossover.", MAX_QUERIES_BEFORE_FREEZE)
            frozen_at_query = MAX_QUERIES_BEFORE_FREEZE
            
        crossover_est = optimizer.get_crossover_estimate()
        logger.info(f"Phase {phase_idx} ({n_docs} docs) Final Crossover: {crossover_est}")
        
        crossover_evolution.append({
            "phase": phase_idx,
            "n_docs": n_docs,
            "crossover_selectivity": crossover_est,
            "queries_to_freeze": frozen_at_query,
            "is_frozen": optimizer.is_frozen
        })
        
        # ── Evaluation Phase (Frozen) ──────────────────────────────────────────
        logger.info("Evaluating Frozen CBO across all selectivity ranges...")
        for f_idx, fspec in enumerate(filters):
            for q_idx in range(NUM_QUERIES):
                est_sel, predict_time_ms = predictor.estimate_timed(fspec)
                predict_overhead_us = predict_time_ms * 1000.0

                t0 = time.perf_counter()
                strategy_name = optimizer.route(est_sel)
                route_overhead_us = (time.perf_counter() - t0) * 1_000_000.0
                
                latency_ms, recall = cache[strategy_name][f_idx][q_idx]
                reward = optimizer.reward_fn.compute(latency_ms, recall)
                
                all_query_details.append(QueryDetail(
                    phase="frozen",
                    n_docs=n_docs,
                    query_idx=q_idx,
                    filter_idx=f_idx,
                    actual_selectivity=fspec.actual_selectivity,
                    estimated_selectivity=est_sel,
                    strategy_chosen=strategy_name,
                    latency_ms=latency_ms,
                    recall=recall,
                    reward=reward,
                    predict_overhead_us=predict_overhead_us,
                    route_overhead_us=route_overhead_us
                ))
        
        # ── Export Selectivity CSV (Frozen Phase Only) ─────────────────────────
        sel_rows = []
        for f_idx, fspec in enumerate(filters):
            sel = fspec.actual_selectivity
            
            pf_lats = [cache["post_filter"][f_idx][q][0] for q in range(NUM_QUERIES)]
            pf_recs = [cache["post_filter"][f_idx][q][1] for q in range(NUM_QUERIES)]
            
            bf_lats = [cache["bitmap_prefilter"][f_idx][q][0] for q in range(NUM_QUERIES)]
            bf_recs = [cache["bitmap_prefilter"][f_idx][q][1] for q in range(NUM_QUERIES)]

            ids_lats = [cache["bitmap_hnsw_prefilter"][f_idx][q][0] for q in range(NUM_QUERIES)]
            ids_recs = [cache["bitmap_hnsw_prefilter"][f_idx][q][1] for q in range(NUM_QUERIES)]
            
            cbo_entries = [d for d in all_query_details if d.phase == "frozen" and d.n_docs == n_docs and d.filter_idx == f_idx]
            cbo_lats = [d.latency_ms for d in cbo_entries]
            cbo_recs = [d.recall for d in cbo_entries]
            
            sel_rows.append({
                "phase": phase_idx,
                "n_docs": n_docs,
                "selectivity_pct": round(sel * 100, 2),
                "cbo_latency_ms": round(np.mean(cbo_lats), 2) if cbo_lats else 0,
                "cbo_recall": round(np.mean(cbo_recs), 4) if cbo_recs else 0,
                "pf_latency_ms": round(np.mean(pf_lats), 2),
                "pf_recall": round(np.mean(pf_recs), 4),
                "bitmap_bf_latency_ms": round(np.mean(bf_lats), 2),
                "bitmap_bf_recall": round(np.mean(bf_recs), 4),
                "idsel_latency_ms": round(np.mean(ids_lats), 2),
                "idsel_recall": round(np.mean(ids_recs), 4),
            })
            
        csv_path = out_dir / f"phase_{phase_idx}_selectivity_frozen.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sel_rows[0].keys())
            writer.writeheader()
            writer.writerows(sel_rows)
            
        with open(out_dir / f"qtable_phase_{phase_idx}.json", "w") as f:
            json.dump(optimizer.get_q_snapshot(), f, indent=2)
            
    # ── Final Exports ──────────────────────────────────────────────────────────
    with open(out_dir / "crossover_evolution.json", "w") as f:
        json.dump(crossover_evolution, f, indent=2)
        
    query_csv_path = out_dir / "all_queries.csv"
    with open(query_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(all_query_details[0]).keys())
        writer.writeheader()
        for d in all_query_details:
            writer.writerow(asdict(d))
            
    logger.info("Done! Incremental Results saved to %s", out_dir)

if __name__ == "__main__":
    main()

"""
main_benchmark.py
=================
Fixed 200k base test.
Shows performance of CBO (Bandit) vs Baselines (IDSelector, PostFilter, Bitmap BF).
"""

import os
import json
import csv
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
from tqdm import tqdm

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec, generate_filters
from strategies import BitmapPreFilter, BitmapHNSWPreFilter, BruteForce, PostFilter

from cbo import ContextualBanditOptimizer, SoftCliffReward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_QUERIES = 50
RANDOM_SEED = 42

@dataclass
class QueryDetail:
    epoch: int
    query_idx: int
    filter_idx: int
    actual_selectivity: float
    estimated_selectivity: float
    strategy_chosen: str
    latency_ms: float
    recall: float
    reward: float

import pickle
from typing import Tuple

def build_corpus(n_docs: int) -> Tuple[np.ndarray, list, FAISSIndex, BitmapIndex]:
    logger.info("Loading full embeddings and metadata from %s …", config.INDEX_DIR)
    
    full_faiss_idx = FAISSIndex.load(config.INDEX_DIR)
    total_available = full_faiss_idx.hnsw_index.ntotal
    full_embeddings = full_faiss_idx._embeddings  # (N, 768)
    
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
    out_dir = config.RESULTS_DIR / f"main_200k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_docs = 200_000
    embeddings, metadata, faiss_idx, bitmap_idx = build_corpus(n_docs)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    logger.info("Generating filters...")
    filters = generate_filters(metadata, config.SELECTIVITY_TARGETS)
    
    logger.info("Pre-computing exact matching ID sets...")
    for f in filters:
        f.matching_ids = f.resolve_bitmap(bitmap_idx)
        f.actual_selectivity = len(f.matching_ids) / n_docs
        
    query_indices = random.sample(range(n_docs), NUM_QUERIES)
    query_embeddings = embeddings[query_indices]
    
    # ── Ground Truth & Baselines ──────────────────────────────────────────────
    logger.info("Computing Ground Truth & Baselines...")
    cache = {
        "post_filter": [[None]*NUM_QUERIES for _ in range(len(filters))],
        "bitmap_hnsw_prefilter": [[None]*NUM_QUERIES for _ in range(len(filters))],
        "bitmap_prefilter": [[None]*NUM_QUERIES for _ in range(len(filters))]
    }
    
    # Instantiate strategies
    brute_force = BruteForce(faiss_idx, metadata)
    post_filter = PostFilter(faiss_idx, metadata, expansion_factor=config.POST_FILTER_EXPANSION)
    bitmap_prefilter = BitmapPreFilter(faiss_idx, bitmap_idx)
    bitmap_hnsw_prefilter = BitmapHNSWPreFilter(faiss_idx, bitmap_idx)
    
    for f_idx, fspec in enumerate(tqdm(filters, desc="Pre-computing")):
        for q_idx in range(NUM_QUERIES):
            q_emb = query_embeddings[q_idx].reshape(1, -1)
            
            bf_result = brute_force.search(q_emb, config.CBO_TOP_K, fspec)
            true_neighbors = bf_result.ids
            
            # PostFilter
            pf_result = post_filter.search(q_emb, config.CBO_TOP_K, fspec)
            pf_recall = len(set(pf_result.ids) & set(true_neighbors)) / config.CBO_TOP_K
            cache["post_filter"][f_idx][q_idx] = (pf_result.total_time_ms, pf_recall)
            
            # IDSelector (BitmapHNSW)
            bh_result = bitmap_hnsw_prefilter.search(q_emb, config.CBO_TOP_K, fspec)
            bh_recall = len(set(bh_result.ids) & set(true_neighbors)) / config.CBO_TOP_K
            cache["bitmap_hnsw_prefilter"][f_idx][q_idx] = (bh_result.total_time_ms, bh_recall)
            
            # Brute-Force Bitmap
            bp_result = bitmap_prefilter.search(q_emb, config.CBO_TOP_K, fspec)
            bp_recall = len(set(bp_result.ids) & set(true_neighbors)) / config.CBO_TOP_K
            cache["bitmap_prefilter"][f_idx][q_idx] = (bp_result.total_time_ms, bp_recall)
            
    # ── Bandit CBO Training ──────────────────────────────────────────────────
    logger.info("Running Bandit CBO...")
    optimizer = ContextualBanditOptimizer(n_corpus=n_docs, mode="softmax")
    
    all_query_details = []
    
    for epoch in range(config.CBO_N_EPOCHS):
        query_pairs = [(f, q) for f in range(len(filters)) for q in range(NUM_QUERIES)]
        random.shuffle(query_pairs)
        
        for f_idx, q_idx in query_pairs:
            fspec = filters[f_idx]
            # Route decision (Estimator is used internally!)
            strategy_name, est_sel = optimizer.route(fspec.matching_ids)
            
            # Actually execute the chosen strategy (we use cache)
            # CBO internally assumes "bitmap_prefilter" or "post_filter"
            # We map "bitmap_prefilter" -> "bitmap_hnsw_prefilter" (IDSelector)
            cache_key = "bitmap_hnsw_prefilter" if strategy_name == "bitmap_prefilter" else "post_filter"
            latency_ms, recall = cache[cache_key][f_idx][q_idx]
            
            reward = optimizer.feedback(est_sel, strategy_name, latency_ms, recall)
            
            all_query_details.append(QueryDetail(
                epoch=epoch,
                query_idx=q_idx,
                filter_idx=f_idx,
                actual_selectivity=fspec.actual_selectivity,
                estimated_selectivity=est_sel,
                strategy_chosen=strategy_name,
                latency_ms=latency_ms,
                recall=recall,
                reward=reward
            ))
            
    # ── Export Selectivity CSV ────────────────────────────────────────────────
    sel_rows = []
    for f_idx, fspec in enumerate(filters):
        sel = fspec.actual_selectivity
        
        pf_lats = [cache["post_filter"][f_idx][q][0] for q in range(NUM_QUERIES)]
        pf_recs = [cache["post_filter"][f_idx][q][1] for q in range(NUM_QUERIES)]
        
        ids_lats = [cache["bitmap_hnsw_prefilter"][f_idx][q][0] for q in range(NUM_QUERIES)]
        ids_recs = [cache["bitmap_hnsw_prefilter"][f_idx][q][1] for q in range(NUM_QUERIES)]
        
        bf_lats = [cache["bitmap_prefilter"][f_idx][q][0] for q in range(NUM_QUERIES)]
        bf_recs = [cache["bitmap_prefilter"][f_idx][q][1] for q in range(NUM_QUERIES)]
        
        cbo_entries = [d for d in all_query_details if d.epoch == config.CBO_N_EPOCHS - 1 and d.filter_idx == f_idx]
        cbo_lats = [d.latency_ms for d in cbo_entries]
        cbo_recs = [d.recall for d in cbo_entries]
        
        sel_rows.append({
            "phase": 1,
            "n_docs": n_docs,
            "selectivity_pct": round(sel * 100, 2),
            "cbo_latency_ms": round(np.mean(cbo_lats), 2) if cbo_lats else 0,
            "cbo_recall": round(np.mean(cbo_recs), 4) if cbo_recs else 0,
            "pf_latency_ms": round(np.mean(pf_lats), 2),
            "pf_recall": round(np.mean(pf_recs), 4),
            "idsel_latency_ms": round(np.mean(ids_lats), 2),
            "idsel_recall": round(np.mean(ids_recs), 4),
            "bitmap_bf_latency_ms": round(np.mean(bf_lats), 2),
            "bitmap_bf_recall": round(np.mean(bf_recs), 4),
        })
        
    csv_path = out_dir / "phase_1_selectivity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sel_rows[0].keys())
        writer.writeheader()
        writer.writerows(sel_rows)
        
    with open(out_dir / "qtable_snapshot.json", "w") as f:
        json.dump(optimizer.get_q_snapshot(), f, indent=2)
        
    logger.info(f"Done! Results in {out_dir}")

if __name__ == "__main__":
    main()

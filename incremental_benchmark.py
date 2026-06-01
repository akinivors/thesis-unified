"""
incremental_benchmark.py
========================
Tests crossover adaptation across different corpus sizes (100k, 200k, 300k, 388k).
"""

import os
import json
import csv
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
from tqdm import tqdm

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec, generate_filters
from strategies import BitmapHNSWPreFilter, PostFilter

from cbo import ContextualBanditOptimizer, SoftCliffReward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_QUERIES = 50
RANDOM_SEED = 99
CORPUS_SIZES = [100_000, 200_000, 300_000, 388_000]

import pickle

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
    out_dir = config.RESULTS_DIR / f"incremental_crossover_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    crossover_evolution = []

    for phase, n_docs in enumerate(CORPUS_SIZES, 1):
        logger.info(f"\n{'='*40}\nPhase {phase}: {n_docs} documents\n{'='*40}")
        embeddings, metadata, faiss_idx, bitmap_idx = build_corpus(n_docs)
        
        logger.info("Generating filters...")
        filters = generate_filters(metadata, config.SELECTIVITY_TARGETS)
        for f in filters:
            f.matching_ids = f.resolve_bitmap(bitmap_idx)
            f.actual_selectivity = len(f.matching_ids) / n_docs
            
        query_indices = random.sample(range(n_docs), NUM_QUERIES)
        query_embeddings = embeddings[query_indices]
        
        post_filter = PostFilter(faiss_idx, metadata, expansion_factor=config.POST_FILTER_EXPANSION)
        bitmap_hnsw_prefilter = BitmapHNSWPreFilter(faiss_idx, bitmap_idx)
        
        logger.info("Running Bandit CBO...")
        optimizer = ContextualBanditOptimizer(n_corpus=n_docs, mode="epsilon_greedy")
        
        for epoch in range(config.CBO_N_EPOCHS):
            query_pairs = [(f, q) for f in range(len(filters)) for q in range(NUM_QUERIES)]
            random.shuffle(query_pairs)
            
            for f_idx, q_idx in query_pairs:
                fspec = filters[f_idx]
                q_emb = query_embeddings[q_idx].reshape(1, -1)
                
                strategy_name, est_sel = optimizer.route(fspec.matching_ids)
                
                if strategy_name == "bitmap_prefilter":
                    res = bitmap_hnsw_prefilter.search(q_emb, config.CBO_TOP_K, fspec)
                else:
                    res = post_filter.search(q_emb, config.CBO_TOP_K, fspec)
                
                # To simulate recall we just pass 1.0 here for the benchmark's routing focus
                # Or calculate true recall if needed. We'll pass 1.0 for speed since we just care about latency crossover
                optimizer.feedback(est_sel, strategy_name, res.total_time_ms, 1.0)
                
        snapshot = optimizer.get_q_snapshot()
        
        # Estimate crossover: where Q_bitmap and Q_post flip
        # Filter out empty buckets to avoid default 1.0 values breaking the logic
        active_buckets = [s for s in snapshot if s['visits'] > 0]
        crossover_est = None
        for i in range(len(active_buckets)-1):
            s1 = active_buckets[i]
            s2 = active_buckets[i+1]
            if (s1['q_bitmap'] > s1['q_post']) and (s2['q_bitmap'] < s2['q_post']):
                crossover_est = s1['upper']
                break
                
        logger.info(f"Phase {phase} Crossover Estimate: {crossover_est}")
        crossover_evolution.append({
            "phase": phase,
            "n_docs": n_docs,
            "crossover_selectivity": crossover_est
        })
        
        with open(out_dir / f"qtable_phase_{phase}.json", "w") as f:
            json.dump(snapshot, f, indent=2)
            
    with open(out_dir / "crossover_evolution.json", "w") as f:
        json.dump(crossover_evolution, f, indent=2)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()

"""
strategies.py — Filtering Strategy Implementations (FAISS)
============================================================
Implements the four query mechanisms under evaluation:

1. **BruteForce**       — Exhaustive (IndexFlatIP) search on ALL vectors.
                          Ground truth baseline for recall computation.
2. **NaivePreFilter**   — Linear scan through metadata to find matching
                          document IDs, then vector search on the subset.
3. **PostFilter**       — HNSW search (no filter, oversampling), then
                          Python-side metadata filtering on results.
4. **BitmapPreFilter**  — Roaring Bitmap lookup (µs) to find matching
                          document IDs, then vector search on the subset.

The key comparison is between (2) and (4): both do the same vector
search on the filtered subset, but the *filtering cost* differs:
  - NaivePreFilter:  O(N) linear scan through metadata dicts
  - BitmapPreFilter: O(1) bitmap set operations

All strategies share the same ``search()`` interface and accept a pre-
computed query embedding so that embedding time is *excluded* from the
benchmark measurements.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

import config
from bitmap_index import BitmapIndex
from faiss_index import FAISSIndex
from filters import FilterSpec

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Search Result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchResult:
    """Container for the output of a single filtered search.

    Attributes
    ----------
    ids : list[str]
        Retrieved document IDs, ordered by relevance.
    distances : list[float]
        Corresponding distance / dissimilarity scores.
    filter_time_ms : float
        Time spent on the filtering stage (ms).
    search_time_ms : float
        Time spent on the vector-search stage (ms).
    total_time_ms : float
        End-to-end wall-clock time for the entire operation (ms).
    candidates_after_filter : int
        Number of candidates remaining after the filter step.
    """

    ids: List[str] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    filter_time_ms: float = 0.0
    search_time_ms: float = 0.0
    total_time_ms: float = 0.0
    candidates_after_filter: int = 0

    @property
    def result_count(self) -> int:
        return len(self.ids)


# ═══════════════════════════════════════════════════════════════════════════════
#  Abstract Base
# ═══════════════════════════════════════════════════════════════════════════════

class FilterStrategy(ABC):
    """Common contract for every filtering strategy."""

    name: str = "base"

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
        selectivity: Optional[float] = None,
    ) -> SearchResult:
        """Execute a filtered top-K search.

        Parameters
        ----------
        query_embedding : np.ndarray
            Pre-computed, un-normalised query vector.
        top_k : int
            Number of results to return.
        filter_spec : FilterSpec
            Metadata filter specification.
        selectivity : float, optional
            Estimated fraction of corpus matching the filter.  Strategies
            that use HNSW with an IDSelector (BitmapHNSWPreFilter) use this
            to adaptively inflate efSearch.  PostFilter uses it to inflate
            the oversampling factor.  Can be left None for strategies that
            don't benefit from it (BruteForce, NaivePreFilter, BitmapPreFilter).
        """


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 — Brute-Force Baseline (Ground Truth)
# ═══════════════════════════════════════════════════════════════════════════════

class BruteForce(FilterStrategy):
    """Exhaustive search on ALL vectors via IndexFlatIP.

    No approximate index is used.  The query vector is compared against
    every document vector, producing the exact top-K.  Metadata filtering
    is applied after the exhaustive search.

    This strategy serves as the **ground truth** for recall computation.
    Its recall is always 1.0 by definition.
    """

    name = "brute_force"

    def __init__(
        self,
        faiss_idx: FAISSIndex,
        all_metadatas: List[Dict[str, Any]],
    ) -> None:
        self.faiss_idx = faiss_idx
        self.all_metadatas = all_metadatas

        # Pre-build the flat index once (avoid re-adding on every query)
        self._flat = faiss.IndexFlatIP(faiss_idx.dim)
        self._flat.add(faiss_idx._embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
        selectivity: Optional[float] = None,
    ) -> SearchResult:
        t0 = time.perf_counter()

        # Exhaustive search on ALL vectors (no HNSW)
        q = np.atleast_2d(query_embedding).astype(np.float32).copy()
        faiss.normalize_L2(q)
        distances, ids = self._flat.search(q, self._flat.ntotal)  # retrieve all

        search_ms = (time.perf_counter() - t0) * 1000

        # Filter results by metadata predicate
        t1 = time.perf_counter()
        filtered_ids: List[str] = []
        filtered_dists: List[float] = []

        for j in range(ids.shape[1]):
            int_id = int(ids[0, j])
            if int_id < 0:
                continue
            if filter_spec.matches(self.all_metadatas[int_id]):
                str_id = self.faiss_idx.int_to_str.get(int_id, f"unknown_{int_id}")
                filtered_ids.append(str_id)
                filtered_dists.append(float(1.0 - distances[0, j]))
                if len(filtered_ids) >= top_k:
                    break

        filter_ms = (time.perf_counter() - t1) * 1000

        return SearchResult(
            ids=filtered_ids,
            distances=filtered_dists,
            filter_time_ms=filter_ms,
            search_time_ms=search_ms,
            total_time_ms=search_ms + filter_ms,
            candidates_after_filter=len(filtered_ids),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 — Naive Pre-Filter (Linear Metadata Scan)
# ═══════════════════════════════════════════════════════════════════════════════

class NaivePreFilter(FilterStrategy):
    """Linear scan through ALL metadata → vector search on matching subset.

    1. Iterate through every metadata dict, check the predicate → O(N).
    2. Collect the integer IDs of matching documents.
    3. Use FAISS IndexFlatIP on only the matching subset for exact search.

    This is the simplest pre-filtering approach.  The filtering cost is
    proportional to the total number of documents regardless of selectivity.
    """

    name = "naive_prefilter"

    def __init__(
        self,
        faiss_idx: FAISSIndex,
        all_metadatas: List[Dict[str, Any]],
    ) -> None:
        self.faiss_idx = faiss_idx
        self.all_metadatas = all_metadatas

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
        selectivity: Optional[float] = None,
    ) -> SearchResult:
        # ── Step 1: Linear scan to find matching IDs ──────────────────────
        t0 = time.perf_counter()
        matching_ids: List[int] = []
        for i, meta in enumerate(self.all_metadatas):
            if filter_spec.matches(meta):
                matching_ids.append(i)
        filter_ms = (time.perf_counter() - t0) * 1000

        n_candidates = len(matching_ids)

        if n_candidates == 0:
            return SearchResult(
                filter_time_ms=filter_ms,
                search_time_ms=0.0,
                total_time_ms=filter_ms,
                candidates_after_filter=0,
            )

        # ── Step 2: Exact vector search on subset ─────────────────────────
        t1 = time.perf_counter()
        subset_int_ids = np.array(matching_ids, dtype=np.int64)
        distances, original_ids = self.faiss_idx.search_flat_subset(
            query_embedding, top_k, subset_int_ids,
        )
        search_ms = (time.perf_counter() - t1) * 1000

        # Map int IDs back to string IDs
        result_ids = []
        result_dists = []
        for i in range(len(original_ids)):
            int_id = int(original_ids[i])
            if int_id < 0:
                continue
            str_id = self.faiss_idx.int_to_str.get(int_id, f"unknown_{int_id}")
            result_ids.append(str_id)
            result_dists.append(float(1.0 - distances[i]))

        return SearchResult(
            ids=result_ids,
            distances=result_dists,
            filter_time_ms=filter_ms,
            search_time_ms=search_ms,
            total_time_ms=filter_ms + search_ms,
            candidates_after_filter=n_candidates,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3 — Post-Filter (Oversampling)
# ═══════════════════════════════════════════════════════════════════════════════

class PostFilter(FilterStrategy):
    """HNSW search without filter → Python-side metadata filtering.

    1. Query FAISS HNSW with an adaptively inflated fetch count (no filter).
    2. For each returned candidate, check the metadata predicate.
    3. Keep the first ``top_k`` candidates that pass the filter.

    The fetch count adapts to selectivity so that the expected number of
    matching candidates is always well above top_k:

        fetch_n = top_k × max(expansion_factor, ceil(2 / selectivity))

    At sel=0.10, top_k=10: fetch_n = 10 × max(100, 20) = 1,000  → expect 100 matches
    At sel=0.01, top_k=10: fetch_n = 10 × max(100, 200) = 2,000 → expect 20 matches
    At sel=0.60, top_k=10: fetch_n = 10 × max(100, 4)   = 1,000 → expect 600 matches

    Risk: if selectivity is extremely low (< 0.001) the required fetch_n
    approaches N, making this strategy degenerate.  The CBO guardrail
    redirects such queries to BitmapPreFilter before they reach here.
    """

    name = "post_filter"

    def __init__(
        self,
        faiss_idx: FAISSIndex,
        all_metadatas: List[Dict[str, Any]],
        expansion_factor: int = config.POST_FILTER_EXPANSION,
    ) -> None:
        self.faiss_idx = faiss_idx
        self.all_metadatas = all_metadatas
        self.expansion_factor = expansion_factor

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
        selectivity: Optional[float] = None,
    ) -> SearchResult:
        # Adaptive expansion: inflate fetch_n when selectivity is low so that
        # the expected number of filtered candidates stays safely above top_k.
        if selectivity is not None and selectivity > 0:
            adaptive_factor = max(self.expansion_factor, int(2.0 / max(selectivity, 1e-3)))
        else:
            adaptive_factor = self.expansion_factor

        fetch_n = min(
            top_k * adaptive_factor,
            self.faiss_idx.hnsw_index.ntotal,
        )

        # ── HNSW search (no filter) ───────────────────────────────────────
        t0 = time.perf_counter()
        distances, ids = self.faiss_idx.search_hnsw(query_embedding, fetch_n)
        search_ms = (time.perf_counter() - t0) * 1000

        # ── Python-side filtering ──────────────────────────────────────────
        t1 = time.perf_counter()
        filtered_ids: List[str] = []
        filtered_dists: List[float] = []

        for j in range(ids.shape[1]):
            int_id = int(ids[0, j])
            if int_id < 0:
                continue
            meta = self.all_metadatas[int_id]
            if filter_spec.matches(meta):
                str_id = self.faiss_idx.int_to_str.get(int_id, f"unknown_{int_id}")
                filtered_ids.append(str_id)
                filtered_dists.append(float(1.0 - distances[0, j]))
                if len(filtered_ids) >= top_k:
                    break

        filter_ms = (time.perf_counter() - t1) * 1000

        return SearchResult(
            ids=filtered_ids,
            distances=filtered_dists,
            filter_time_ms=filter_ms,
            search_time_ms=search_ms,
            total_time_ms=search_ms + filter_ms,
            candidates_after_filter=len(filtered_ids),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 4 — Bitmap Pre-Filter (Roaring Bitmap + Subset Search)
# ═══════════════════════════════════════════════════════════════════════════════

class BitmapPreFilter(FilterStrategy):
    """Roaring Bitmap lookup → exact brute-force vector search on subset.

    1. Resolve matching document IDs via bitmap set operations (µs).
    2. Use FAISS IndexFlatIP on only the matching vectors for exact search.

    Compared to NaivePreFilter, the filtering step is orders of magnitude
    faster (bitmap set-intersection vs linear predicate scan).  The vector
    search step is identical.

    This strategy is one of the two CBO arms (vs PostFilter).  Its latency
    scales linearly with |filtered subset|, so it wins at low selectivity
    and loses at high selectivity — exactly the crossover the CBO learns.
    """

    name = "bitmap_prefilter"

    def __init__(
        self,
        faiss_idx: FAISSIndex,
        bitmap_index: BitmapIndex,
    ) -> None:
        self.faiss_idx = faiss_idx
        self.bitmap_index = bitmap_index

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
        selectivity: Optional[float] = None,  # unused; brute-force needs no ef tuning
    ) -> SearchResult:
        # ── Step 1: Bitmap resolution (µs-level) ──────────────────────────
        t0 = time.perf_counter()
        matching_bitmap = filter_spec.resolve_bitmap(self.bitmap_index)
        matching_ids = np.array(matching_bitmap.to_array(), dtype=np.int64)
        filter_ms = (time.perf_counter() - t0) * 1000

        n_candidates = len(matching_ids)

        if n_candidates == 0:
            return SearchResult(
                filter_time_ms=filter_ms,
                search_time_ms=0.0,
                total_time_ms=filter_ms,
                candidates_after_filter=0,
            )

        # ── Step 2: Exact brute-force on subset ──────────────────────────
        t1 = time.perf_counter()
        distances, original_ids = self.faiss_idx.search_flat_subset(
            query_embedding, top_k, matching_ids,
        )
        search_ms = (time.perf_counter() - t1) * 1000

        # Map int IDs back to string IDs
        result_ids = []
        result_dists = []
        for i in range(len(original_ids)):
            int_id = int(original_ids[i])
            if int_id < 0:
                continue
            str_id = self.faiss_idx.int_to_str.get(int_id, f"unknown_{int_id}")
            result_ids.append(str_id)
            result_dists.append(float(1.0 - distances[i]))

        return SearchResult(
            ids=result_ids,
            distances=result_dists,
            filter_time_ms=filter_ms,
            search_time_ms=search_ms,
            total_time_ms=filter_ms + search_ms,
            candidates_after_filter=n_candidates,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 5 — Bitmap + HNSW IDSelector Pre-Filter
# ═══════════════════════════════════════════════════════════════════════════════

class BitmapHNSWPreFilter(FilterStrategy):
    """Roaring Bitmap lookup → HNSW search with IDSelector constraint.

    1. Resolve matching document IDs via bitmap set operations (µs).
    2. Build a FAISS IDSelectorBitmap from the matching IDs (memory-efficient,
       O(1) bit-check vs IDSelectorBatch's hash-set with O(1) avg + overhead).
    3. Search HNSW with the IDSelector and adaptive efSearch — the graph is
       traversed normally but only nodes passing the selector are returned.

    efSearch is inflated proportionally to 1/selectivity so that the HNSW
    traversal explores enough nodes to reach the sparse filtered subgraph.
    Without this, ef=128 fails catastrophically at sel < 0.05.

    Note: this strategy is NOT a CBO arm — it is a benchmark baseline.
    The CBO routes between BitmapPreFilter (exact) and PostFilter (approx).
    """

    name = "bitmap_hnsw_prefilter"

    def __init__(
        self,
        faiss_idx: FAISSIndex,
        bitmap_index: BitmapIndex,
    ) -> None:
        self.faiss_idx = faiss_idx
        self.bitmap_index = bitmap_index

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filter_spec: FilterSpec,
        selectivity: Optional[float] = None,
    ) -> SearchResult:
        # ── Step 1: Bitmap resolution (µs-level) ──────────────────────────
        t0 = time.perf_counter()
        matching_bitmap = filter_spec.resolve_bitmap(self.bitmap_index)
        matching_ids = np.array(matching_bitmap.to_array(), dtype=np.int64)
        filter_ms = (time.perf_counter() - t0) * 1000

        n_candidates = len(matching_ids)

        if n_candidates == 0:
            return SearchResult(
                filter_time_ms=filter_ms,
                search_time_ms=0.0,
                total_time_ms=filter_ms,
                candidates_after_filter=0,
            )

        # ── Step 2: HNSW search with IDSelectorBitmap ────────────────────
        # IDSelectorBitmap: raw bit array — O(1) check, 64x less memory than
        # IDSelectorBatch (50KB vs 3.2MB for 400K docs, cache-friendly).
        t1 = time.perf_counter()
        N = self.faiss_idx.hnsw_index.ntotal
        bm = np.zeros((N + 7) // 8, dtype=np.uint8)
        for idx in matching_ids:
            bm[idx >> 3] |= np.uint8(1 << (idx & 7))
        id_selector = faiss.IDSelectorBitmap(N, faiss.swig_ptr(bm))

        distances, ids = self.faiss_idx.search_hnsw(
            query_embedding, top_k,
            id_selector=id_selector,
            selectivity=selectivity,  # triggers adaptive ef inflation
        )
        # bm must stay alive until after search() returns
        del bm, id_selector
        search_ms = (time.perf_counter() - t1) * 1000

        # Map int IDs back to string IDs
        result_ids = []
        result_dists = []
        for j in range(ids.shape[1]):
            int_id = int(ids[0, j])
            if int_id < 0:
                continue
            str_id = self.faiss_idx.int_to_str.get(int_id, f"unknown_{int_id}")
            result_ids.append(str_id)
            result_dists.append(float(1.0 - distances[0, j]))

        return SearchResult(
            ids=result_ids,
            distances=result_dists,
            filter_time_ms=filter_ms,
            search_time_ms=search_ms,
            total_time_ms=filter_ms + search_ms,
            candidates_after_filter=n_candidates,
        )


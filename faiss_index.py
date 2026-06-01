"""
faiss_index.py — FAISS Index Management
=========================================
Wraps FAISS HNSW and Flat (brute-force) indices.

Responsibilities:
  - Build and persist a HNSW index for approximate nearest-neighbor search.
  - Build an on-the-fly Flat index for exact brute-force search on subsets.
  - Provide search methods with optional IDSelector for pre-filtering.
  - Manage int64 ⇆ string ID mappings.

All vectors are L2-normalised before insertion so that inner-product
similarity equals cosine similarity.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

import config

log = logging.getLogger(__name__)


class FAISSIndex:
    """Manages a FAISS HNSW index with ID mapping.

    Parameters
    ----------
    dim : int
        Vector dimensionality (default from config).
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self.dim = dim

        # ── HNSW index (inner product → cosine on normalised vecs) ────────
        self.hnsw_index: Optional[faiss.Index] = None

        # ── ID mappings ───────────────────────────────────────────────────
        self.int_to_str: Dict[int, str] = {}   # faiss int64 → doc string ID
        self.str_to_int: Dict[str, int] = {}   # doc string ID → faiss int64

        # ── Raw normalised embeddings (for brute-force fallback) ──────────
        self._embeddings: Optional[np.ndarray] = None

    # ══════════════════════════════════════════════════════════════════════
    #  Index Construction
    # ══════════════════════════════════════════════════════════════════════

    def build(
        self,
        embeddings: np.ndarray,
        doc_ids: List[str],
    ) -> None:
        """Build a HNSW index from embeddings.

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, dim)
            Document embeddings (will be L2-normalised in place).
        doc_ids : list[str]
            Corresponding document IDs (same order as rows in *embeddings*).
        """
        n = len(doc_ids)
        assert embeddings.shape == (n, self.dim), (
            f"Shape mismatch: embeddings {embeddings.shape} vs n={n}, dim={self.dim}"
        )

        # L2-normalise → inner product == cosine similarity
        faiss.normalize_L2(embeddings)
        self._embeddings = embeddings.copy()

        # Integer IDs: 0, 1, 2, …, N-1
        int_ids = np.arange(n, dtype=np.int64)
        for i, doc_id in enumerate(doc_ids):
            self.int_to_str[i] = doc_id
            self.str_to_int[doc_id] = i

        # ── Build HNSW index ──────────────────────────────────────────────
        log.info(
            "Building HNSW index: n=%d, dim=%d, M=%d, efConstruction=%d",
            n, self.dim, config.HNSW_M, config.HNSW_EF_CONSTR,
        )
        hnsw = faiss.IndexHNSWFlat(self.dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = config.HNSW_EF_CONSTR
        hnsw.hnsw.efSearch = config.HNSW_EF_SEARCH

        # IndexHNSWFlat doesn't support add_with_ids, so we wrap it
        # with IndexIDMap to allow ID-based operations if needed.
        # However, HNSW sequential IDs (0..N-1) match our int_to_str mapping,
        # so we add directly.
        hnsw.add(embeddings)
        self.hnsw_index = hnsw

        log.info("HNSW index built: ntotal=%d", self.hnsw_index.ntotal)

    # ══════════════════════════════════════════════════════════════════════
    #  Search Methods
    # ══════════════════════════════════════════════════════════════════════

    def search_hnsw(
        self,
        query: np.ndarray,
        top_k: int,
        id_selector: Optional[faiss.IDSelector] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the HNSW index, optionally restricting to a set of IDs.

        Parameters
        ----------
        query : np.ndarray, shape (1, dim) or (dim,)
            Query vector (will be normalised).
        top_k : int
            Number of results.
        id_selector : faiss.IDSelector, optional
            If provided, only IDs accepted by this selector are considered.

        Returns
        -------
        (distances, ids) : tuple of np.ndarray
            distances shape (1, top_k), ids shape (1, top_k).
            IDs are int64 FAISS IDs. Use ``int_to_str`` to map back.
        """
        q = np.atleast_2d(query).astype(np.float32).copy()
        faiss.normalize_L2(q)

        if id_selector is not None:
            params = faiss.SearchParametersHNSW()
            params.sel = id_selector
            params.efSearch = config.HNSW_EF_SEARCH
            distances, ids = self.hnsw_index.search(q, top_k, params=params)
        else:
            distances, ids = self.hnsw_index.search(q, top_k)

        return distances, ids

    def search_flat_subset(
        self,
        query: np.ndarray,
        top_k: int,
        subset_int_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact brute-force search over a subset of vectors.

        Builds a temporary IndexFlatIP, copies only the subset embeddings
        into it, searches, and maps IDs back.

        Parameters
        ----------
        query : np.ndarray, shape (dim,)
            Query vector (will be normalised).
        top_k : int
            Number of results.
        subset_int_ids : np.ndarray of int64
            The integer IDs of vectors to include.

        Returns
        -------
        (distances, original_ids) : tuple of np.ndarray
            distances shape (top_k,), original_ids shape (top_k,).
        """
        q = np.atleast_2d(query).astype(np.float32).copy()
        faiss.normalize_L2(q)

        # Gather subset embeddings
        subset_vecs = self._embeddings[subset_int_ids]  # (m, dim)

        # Build temporary flat index
        flat = faiss.IndexFlatIP(self.dim)
        flat.add(subset_vecs)

        k = min(top_k, len(subset_int_ids))
        distances, local_ids = flat.search(q, k)

        # Map local flat-index IDs back to original int IDs
        original_ids = subset_int_ids[local_ids[0]]
        return distances[0], original_ids

    # ══════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════

    def save(self, directory: Path) -> None:
        """Save the HNSW index and metadata to *directory*."""
        directory.mkdir(parents=True, exist_ok=True)

        # FAISS binary index
        index_path = directory / "hnsw.index"
        faiss.write_index(self.hnsw_index, str(index_path))
        log.info("FAISS HNSW index saved → %s", index_path)

        # Embeddings (for brute-force fallback)
        emb_path = directory / "embeddings.npy"
        np.save(emb_path, self._embeddings)
        log.info("Embeddings saved → %s", emb_path)

        # ID mappings
        map_path = directory / "id_mappings.pkl"
        with open(map_path, "wb") as fh:
            pickle.dump(
                {"int_to_str": self.int_to_str, "str_to_int": self.str_to_int},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log.info("ID mappings saved → %s", map_path)

    @classmethod
    def load(cls, directory: Path) -> "FAISSIndex":
        """Load a previously saved index."""
        idx = cls()

        # FAISS index
        index_path = directory / "hnsw.index"
        idx.hnsw_index = faiss.read_index(str(index_path))
        idx.dim = idx.hnsw_index.d
        log.info(
            "FAISS HNSW index loaded ← %s  (ntotal=%d)",
            index_path, idx.hnsw_index.ntotal,
        )

        # Embeddings
        emb_path = directory / "embeddings.npy"
        idx._embeddings = np.load(emb_path)
        log.info("Embeddings loaded ← %s  shape=%s", emb_path, idx._embeddings.shape)

        # ID mappings
        map_path = directory / "id_mappings.pkl"
        with open(map_path, "rb") as fh:
            maps = pickle.load(fh)
        idx.int_to_str = maps["int_to_str"]
        idx.str_to_int = maps["str_to_int"]
        log.info("ID mappings loaded ← %s  (%d docs)", map_path, len(idx.int_to_str))

        return idx

"""
selectivity_predictor.py
========================
Mini-Bitmap Selectivity Predictor.

Builds a small BitmapIndex from a random subset (default 3%) of the full
metadata corpus. Selectivity estimation is performed by running the same
filter on this mini-bitmap and dividing by the sample size.

This module is **completely CBO-independent** and can be used standalone.

Key properties:
  - O(µs) estimation time via Roaring Bitmap intersection on a tiny set.
  - Compound filter support (AND, OR, range) — correlation preserved in sample.
  - Rebuild trigger when corpus size changes by more than a configurable threshold.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pyroaring import BitMap

from bitmap_index import BitmapIndex
import config

log = logging.getLogger(__name__)


class MiniBitmapPredictor:
    """Estimates filter selectivity using a sampled mini-bitmap index.

    Parameters
    ----------
    metadata : list[dict]
        Full metadata list from the corpus.
    sample_ratio : float
        Fraction of corpus to sample (default 0.03 = 3%).
    seed : int
        Random seed for reproducible sampling.
    rebuild_threshold : float
        Fraction of corpus size change that triggers a rebuild (default 0.30).
    """

    def __init__(
        self,
        metadata: List[Dict[str, Any]],
        sample_ratio: float = 0.03,
        seed: int = 42,
        rebuild_threshold: float = 0.30,
    ) -> None:
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.rebuild_threshold = rebuild_threshold
        self.N_at_build: int = 0
        self.sample_size: int = 0
        self._mini_bitmap: Optional[BitmapIndex] = None
        # Maps sample-local index -> original corpus index
        self._sample_indices: Optional[np.ndarray] = None

        self._build(metadata)

    # ── Construction ───────────────────────────────────────────────────────

    def _build(self, metadata: List[Dict[str, Any]]) -> None:
        """Build the mini-bitmap index from a random sample of metadata."""
        self.N_at_build = len(metadata)
        self.sample_size = max(1, int(self.N_at_build * self.sample_ratio))

        rng = np.random.default_rng(self.seed)
        self._sample_indices = rng.choice(
            self.N_at_build, size=self.sample_size, replace=False
        )
        self._sample_indices.sort()  # sorted for cache-friendly access

        self._mini_bitmap = BitmapIndex()
        for local_idx, global_idx in enumerate(self._sample_indices):
            doc_meta = metadata[global_idx]
            self._mini_bitmap.add_document(
                local_idx, str(local_idx), doc_meta
            )

        log.info(
            "MiniBitmapPredictor built: %d/%d docs sampled (%.1f%%), "
            "%d fields indexed",
            self.sample_size,
            self.N_at_build,
            self.sample_ratio * 100,
            len(self._mini_bitmap.field_bitmaps),
        )

    def rebuild(self, metadata: List[Dict[str, Any]]) -> None:
        """Rebuild the mini-bitmap with fresh metadata."""
        log.info(
            "Rebuilding MiniBitmapPredictor: N changed %d -> %d",
            self.N_at_build,
            len(metadata),
        )
        self._build(metadata)

    # ── Estimation ─────────────────────────────────────────────────────────

    def estimate(self, filter_spec) -> float:
        """Estimate selectivity of a FilterSpec using the mini-bitmap.

        Parameters
        ----------
        filter_spec : FilterSpec
            Must have a ``bitmap_resolver`` callable that accepts a BitmapIndex.

        Returns
        -------
        float
            Estimated selectivity in [0.0, 1.0].
        """
        matching = filter_spec.resolve_bitmap(self._mini_bitmap)
        return len(matching) / self.sample_size if self.sample_size > 0 else 0.0

    def estimate_timed(self, filter_spec) -> Tuple[float, float]:
        """Estimate selectivity and return (selectivity, elapsed_microseconds).

        Returns
        -------
        tuple[float, float]
            (estimated_selectivity, estimation_time_us)
        """
        t0 = time.perf_counter()
        sel = self.estimate(filter_spec)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        return sel, elapsed_us

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def needs_rebuild(self, current_N: int) -> bool:
        """Check if the corpus has changed enough to warrant a rebuild.

        Returns True if corpus size changed by more than ``rebuild_threshold``
        (default 30%) since the last build.
        """
        if self.N_at_build == 0:
            return True
        change_ratio = abs(current_N - self.N_at_build) / self.N_at_build
        return change_ratio > self.rebuild_threshold

    # ── Introspection ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the predictor."""
        return {
            "N_at_build": self.N_at_build,
            "sample_size": self.sample_size,
            "sample_ratio": self.sample_ratio,
            "rebuild_threshold": self.rebuild_threshold,
            "n_fields": len(self._mini_bitmap.field_bitmaps),
            "field_cardinalities": {
                field: len(val_map)
                for field, val_map in self._mini_bitmap.field_bitmaps.items()
            },
        }

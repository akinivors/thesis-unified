"""
estimator.py
============
Selectivity estimation via reservoir sampling.
Instead of scanning the full corpus, maintains a random reservoir.
"""

import numpy as np

class SelectivityEstimator:
    """
    Estimates filter selectivity via reservoir sampling.
    """

    RESERVOIR_SIZE = 2_000

    def __init__(self, N: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.reservoir = rng.integers(0, N, size=self.RESERVOIR_SIZE)

    def estimate(self, candidate_set: set[int]) -> float:
        """
        Returns estimated selectivity in [0, 1].
        candidate_set: set of global indices that pass the filter.
        """
        hits = sum(1 for idx in self.reservoir if idx in candidate_set)
        return hits / self.RESERVOIR_SIZE

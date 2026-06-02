"""
qtable.py
=========
Discretized Q-table with variable-granularity buckets (Phase 1 & Phase 2).
"""

import threading
from typing import List, Tuple, Dict
import config

class QTable:
    """
    Phase 1: uniform buckets.
    Phase 2: adaptive resampling concentrates tight buckets around the battleground.
    """

    ARMS = ("bitmap_prefilter", "post_filter")

    def __init__(self, sigma_lower: float, sigma_upper: float):
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.phase       = 1
        self._lock       = threading.Lock()

        # Hardcode phase widths or read from config if defined
        self.bucket_width_phase1 = getattr(config, "BUCKET_WIDTH_PHASE1", 0.02)
        self.bucket_width_battle = getattr(config, "BUCKET_WIDTH_BATTLE", 0.005)
        self.bucket_width_extreme = getattr(config, "BUCKET_WIDTH_EXTREME", 0.10)
        self.theta_split = getattr(config, "THETA_SPLIT", 0.05)
        self.m_max_buckets = getattr(config, "M_MAX_BUCKETS", 50)
        self.alpha_0 = getattr(config, "CBO_ALPHA", 0.3)
        self.alpha_min = getattr(config, "ALPHA_MIN", 0.01)
        self.lr_decay_rate = getattr(config, "LR_DECAY_RATE", 0.01)

        self._build_uniform_buckets()

    def _build_uniform_buckets(self):
        lo = self.sigma_lower
        hi = self.sigma_upper
        width = self.bucket_width_phase1

        boundaries = []
        cur = lo
        while cur < hi - 1e-9:
            nxt = min(cur + width, hi)
            boundaries.append((cur, nxt))
            cur = nxt

        self._init_from_boundaries(boundaries)

    def _init_from_boundaries(self, boundaries: List[Tuple[float, float]]):
        self.buckets = boundaries
        self.Q = {
            i: {arm: 1.0 for arm in self.ARMS}
            for i in range(len(boundaries))
        }
        self.N = {i: 0 for i in range(len(boundaries))}

    def bucket_id(self, selectivity: float) -> int:
        s = max(self.sigma_lower, min(selectivity, self.sigma_upper - 1e-9))
        for i, (lo, hi) in enumerate(self.buckets):
            if lo <= s < hi:
                return i
        return len(self.buckets) - 1

    def get_q_values(self, selectivity: float) -> Dict[str, float]:
        bucket_idx = self.bucket_id(selectivity)
        with self._lock:
            return dict(self.Q[bucket_idx])

    def get_visits(self, selectivity: float) -> int:
        bucket_idx = self.bucket_id(selectivity)
        with self._lock:
            return self.N[bucket_idx]

    def update(self, selectivity: float, arm: str, reward: float) -> None:
        bucket_idx = self.bucket_id(selectivity)
        with self._lock:
            self.N[bucket_idx] += 1
            n = self.N[bucket_idx]
            alpha = max(self.alpha_min, self.alpha_0 / (1 + self.lr_decay_rate * n))
            self.Q[bucket_idx][arm] += alpha * (reward - self.Q[bucket_idx][arm])

    def resample_buckets(self) -> None:
        with self._lock:
            if self.phase == 2:
                return # Already resampled
            self.phase = 2

            battleground_mid = []
            for i, (lo, hi) in enumerate(self.buckets):
                q = self.Q[i]
                if self.N[i] > 0 and abs(q[self.ARMS[0]] - q[self.ARMS[1]]) < self.theta_split:
                    battleground_mid.append((lo + hi) / 2)

            if not battleground_mid:
                return

            bg_lo = max(self.sigma_lower, min(battleground_mid) - 0.05)
            bg_hi = min(self.sigma_upper, max(battleground_mid) + 0.05)

            new_boundaries = []
            cur = self.sigma_lower
            while cur < self.sigma_upper - 1e-9:
                if bg_lo <= cur < bg_hi:
                    width = self.bucket_width_battle
                else:
                    width = self.bucket_width_extreme
                    if cur < bg_lo and cur + width > bg_lo:
                        width = bg_lo - cur
                    elif cur >= bg_hi and cur + width > self.sigma_upper:
                        width = self.sigma_upper - cur
                
                nxt = min(cur + width, self.sigma_upper)
                if nxt <= cur:
                    break
                new_boundaries.append((cur, nxt))
                cur = nxt
                if len(new_boundaries) >= self.m_max_buckets:
                    if cur < self.sigma_upper:
                        new_boundaries.append((cur, self.sigma_upper))
                    break

            old_Q = self.Q
            old_N = self.N
            old_buckets = self.buckets
            self._init_from_boundaries(new_boundaries)

            accum = {i: {arm: 0.0 for arm in self.ARMS} for i in range(len(new_boundaries))}
            for i in accum:
                accum[i]["q_count"] = 0
                accum[i]["n_sum"] = 0

            for old_i, (lo, hi) in enumerate(old_buckets):
                if old_N[old_i] == 0:
                    continue
                mid = (lo + hi) / 2
                new_i = self.bucket_id(mid)
                for arm in self.ARMS:
                    accum[new_i][arm] += old_Q[old_i][arm]
                accum[new_i]["q_count"] += 1
                accum[new_i]["n_sum"] += old_N[old_i]

            for new_i, acc in accum.items():
                if acc["q_count"] > 0:
                    for arm in self.ARMS:
                        self.Q[new_i][arm] = acc[arm] / acc["q_count"]
                    self.N[new_i] = acc["n_sum"]

    def check_freeze_condition(
        self,
        min_visits: int = config.CBO_FREEZE_MIN_VISITS,
        max_gap: int = config.CBO_FREEZE_MAX_GAP,
    ) -> float | None:
        """
        Check for a stable bitmap→post crossover across visited buckets.

        Original behaviour (max_gap=0): requires strictly adjacent visited
        buckets with a sign flip.

        Extended behaviour (max_gap>0): allows up to ``max_gap`` consecutively
        unvisited Phase-2 buckets between the last bitmap-winning and the first
        post-winning visited bucket.  This handles training filter sets that
        leave a selectivity gap in the battleground region (e.g. filters jump
        from 9 % to 12 %, leaving 10–11.5 % Phase-2 buckets permanently empty).

        Returns the upper bound of the last bitmap-winning visited bucket, i.e.
        the learned crossover threshold θ*.
        """
        with self._lock:
            # Collect indices of sufficiently visited buckets in order.
            visited = [i for i in range(len(self.buckets)) if self.N[i] >= min_visits]
            last_crossover = None

            for pos in range(len(visited) - 1):
                i = visited[pos]
                j = visited[pos + 1]

                # Number of unvisited buckets between i and j.
                gap = j - i - 1
                if gap > max_gap:
                    continue

                diff_i = self.Q[i][self.ARMS[0]] - self.Q[i][self.ARMS[1]]   # bitmap - post @ i
                diff_j = self.Q[j][self.ARMS[0]] - self.Q[j][self.ARMS[1]]   # bitmap - post @ j

                # Sign flip: bitmap wins at i, post wins at j.
                # Keep updating last_crossover — we want the LAST such flip, not
                # the first.  A non-monotone reward landscape (bitmap better at
                # e.g. 21 %, post better at 18 %, bitmap again at 21–24 %, post
                # permanently from 26 %+) means the first crossing is a local
                # dip; only after the final crossing does post stay dominant.
                if diff_i > 0 and diff_j < 0:
                    last_crossover = self.buckets[i][1]

            return last_crossover

    def get_snapshot(self) -> List[Dict]:
        with self._lock:
            return [
                {
                    "lower": b[0],
                    "upper": b[1],
                    "visits": self.N[i],
                    "q_bitmap": self.Q[i][self.ARMS[0]],
                    "q_post": self.Q[i][self.ARMS[1]]
                }
                for i, b in enumerate(self.buckets)
            ]

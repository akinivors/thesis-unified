"""
optimizer.py
============
Contextual Bandit Optimizer (CBO) decision engine.
Combines Guardrails, QTable, Estimator, and Reward tracking.
Now includes Hybrid Architecture: Continual Learning & Abrupt Reset.
"""

import math
import random
from typing import Tuple, Dict, Any, Optional

import config
from .qtable import QTable
from .estimator import SelectivityEstimator
from .reward import SoftCliffReward


class ContextualBanditOptimizer:
    """
    Decides whether to route to Pre-filter (bitmap/idselector) or Post-filter.
    Supports both 'softmax' and 'epsilon_greedy' exploration.
    Includes Hybrid Continuous Learning and Abrupt Drift detection.
    """
    
    def __init__(self, n_corpus: int, mode: str = "softmax", continual_epsilon: float = 0.01):
        self.n_corpus = n_corpus
        self.mode = mode
        self.continual_epsilon = continual_epsilon
        self.n_corpus_at_freeze = n_corpus
        
        self.sigma_lower = config.CBO_SIGMA_LOWER
        self.sigma_upper = config.CBO_SIGMA_UPPER
        
        self.qtable = QTable(self.sigma_lower, self.sigma_upper)
        self.estimator = SelectivityEstimator(N=n_corpus)
        self.reward_fn = SoftCliffReward(
            l_max=config.CBO_L_MAX, 
            r_target=config.CBO_R_TARGET, 
            beta=config.CBO_BETA,
            margin=getattr(config, "CBO_RECALL_MARGIN", 0.01)
        )
        
        self.tau = getattr(config, "CBO_TAU_INIT", 0.05)
        self.epsilon_max = getattr(config, "CBO_EPSILON_MAX", 0.3)
        self.epsilon_k = getattr(config, "CBO_DECAY_K", 5.0)
        self.n_min_visits = getattr(config, "CBO_FREEZE_MIN_VISITS", 15)
        self.n_warmup = getattr(config, "N_WARMUP", 2000)
        
        self.query_count = 0

        # Freeze State
        self.is_frozen = False
        self.frozen_crossover_point: Optional[float] = None
        self._is_exploring_now = False

        # Phase 1 fallback (Fix 3):
        # Track the last non-None crossover estimate produced while still in
        # Phase 1 (query_count < n_warmup).  If Phase 2 never detects a freeze
        # (e.g. because a filter-gap leaves battleground buckets empty), callers
        # can invoke apply_phase1_fallback() to use this as the frozen θ*.
        self._last_phase1_crossover: Optional[float] = None

    def update_corpus_size(self, new_n_corpus: int) -> bool:
        """
        Triggered when dataset size changes.
        If size changes > 30%, triggers an abrupt unfreeze.
        Returns True if unfroze.
        """
        self.n_corpus = new_n_corpus
        self.estimator.N = new_n_corpus
        if self.is_frozen:
            change_ratio = abs(self.n_corpus - self.n_corpus_at_freeze) / self.n_corpus_at_freeze
            if change_ratio > 0.30:
                self.unfreeze(reason=f"Size changed by {change_ratio*100:.1f}%")
                return True
        return False

    def unfreeze(self, reason: str = "Manual Trigger"):
        """Completely reset the freeze state and slightly soften the Q-table to relearn."""
        self.is_frozen = False
        self.frozen_crossover_point = None
        self.n_corpus_at_freeze = self.n_corpus
        # Soften Q-Table to encourage re-exploration without fully erasing memory
        with self.qtable._lock:
            for i in range(len(self.qtable.buckets)):
                self.qtable.Q[i]["bitmap_prefilter"] = (self.qtable.Q[i]["bitmap_prefilter"] + 1.0) / 2.0
                self.qtable.Q[i]["post_filter"] = (self.qtable.Q[i]["post_filter"] + 1.0) / 2.0
                self.qtable.N[i] = max(0, self.qtable.N[i] // 2)

    def route(self, selectivity: float) -> str:
        """
        Returns strategy_name based on selectivity.
        If frozen, uses the learned crossover point directly (with 1% explore chance).
        """
        self._is_exploring_now = False

        if selectivity < self.sigma_lower:
            return "bitmap_prefilter"
        if selectivity > self.sigma_upper:
            return "post_filter"

        if self.is_frozen and self.frozen_crossover_point is not None:
            # 1% Continual Learning check
            if random.random() < self.continual_epsilon:
                self._is_exploring_now = True
                return "bitmap_prefilter" if random.random() < 0.5 else "post_filter"
            else:
                return "bitmap_prefilter" if selectivity < self.frozen_crossover_point else "post_filter"
            
        n_visits = self.qtable.get_visits(selectivity)
        if n_visits < self.n_min_visits:
            self._is_exploring_now = True
            return "bitmap_prefilter" if random.random() < 0.5 else "post_filter"
            
        self._is_exploring_now = True
        q_vals = self.qtable.get_q_values(selectivity)
        
        if self.mode == "softmax":
            return self._softmax_choice(q_vals)
        else:
            return self._epsilon_greedy_choice(q_vals)

    def _softmax_choice(self, q: dict) -> str:
        q_pre = q["bitmap_prefilter"]
        q_post = q["post_filter"]
        max_q = max(q_pre, q_post)
        e_pre = math.exp((q_pre - max_q) / self.tau)
        e_post = math.exp((q_post - max_q) / self.tau)
        p_pre = e_pre / (e_pre + e_post)
        return "bitmap_prefilter" if random.random() < p_pre else "post_filter"

    def _epsilon_greedy_choice(self, q: dict) -> str:
        q_pre = q["bitmap_prefilter"]
        q_post = q["post_filter"]
        delta = abs(q_pre - q_post)
        epsilon = self.epsilon_max * math.exp(-self.epsilon_k * delta)
        if random.random() < epsilon:
            return "bitmap_prefilter" if random.random() < 0.5 else "post_filter"
        return "bitmap_prefilter" if q_pre >= q_post else "post_filter"

    def feedback(self, selectivity: float, strategy_name: str, latency_ms: float, recall: float) -> float:
        """
        Computes reward and updates Q-table if exploring.
        Detects Content Drift if Q-table flips while frozen.
        """
        reward = self.reward_fn.compute(latency_ms, recall)
        
        # Only update if we were actually exploring
        if self._is_exploring_now and self.sigma_lower <= selectivity <= self.sigma_upper:
            self.qtable.update(selectivity, strategy_name, reward)
            
            new_crossover = self.qtable.check_freeze_condition(min_visits=self.n_min_visits)

            # Fix 3: record the last Phase 1 crossover before Phase 2 fires.
            # Phase 1 uses coarser buckets that can still span the filter gap,
            # so its crossover estimate is the best available fallback when
            # Phase 2 battleground buckets are left permanently unvisited.
            if self.query_count < self.n_warmup and new_crossover is not None:
                self._last_phase1_crossover = new_crossover

            if not self.is_frozen:
                # Only allow freeze after N_WARMUP queries so Phase-2
                # resampling (triggered at query_count == n_warmup) has
                # already fired and all buckets have been visited.
                # query_count is incremented *after* this block, so the
                # first time this guard is True is on query n_warmup+1.
                if new_crossover is not None and self.query_count >= self.n_warmup:
                    self.is_frozen = True
                    self.frozen_crossover_point = new_crossover
                    self.n_corpus_at_freeze = self.n_corpus
            else:
                # We are frozen, but 1% continual learning updated the Q-table.
                # Check if the leadership at the frozen crossover point changed!
                if new_crossover is None or abs(new_crossover - self.frozen_crossover_point) > 0.05:
                    self.unfreeze(reason="Content Drift Detected (Q-table flipped)")
            
        self.query_count += 1
        
        if not self.is_frozen and self.query_count == self.n_warmup:
            self.qtable.resample_buckets()
            
        return reward

    def get_q_snapshot(self) -> list:
        return self.qtable.get_snapshot()

    def get_crossover_estimate(self) -> Optional[float]:
        if self.is_frozen:
            return self.frozen_crossover_point
        return self.qtable.check_freeze_condition()

    def apply_phase1_fallback(self) -> bool:
        """
        Apply the last Phase 1 crossover estimate as the frozen θ* when
        Phase 2 failed to detect a freeze on its own.

        This handles the case where fine-grained Phase 2 battleground buckets
        are left permanently unvisited due to a gap in the training filter
        set's selectivity distribution.  Phase 1's coarser (0.02-width) buckets
        span the gap and produce a valid crossover estimate; that estimate is
        stored in ``_last_phase1_crossover`` during training and applied here.

        Returns True if the fallback was applied, False if it was not needed
        (already frozen) or not available (no Phase 1 crossover was ever seen).
        """
        if self.is_frozen:
            return False  # Phase 2 succeeded; nothing to do
        if self._last_phase1_crossover is None:
            return False  # No Phase 1 estimate was ever recorded
        if self.query_count < self.n_warmup:
            return False  # Phase 2 hasn't run yet; too early to fall back

        self.is_frozen = True
        self.frozen_crossover_point = self._last_phase1_crossover
        self.n_corpus_at_freeze = self.n_corpus
        return True

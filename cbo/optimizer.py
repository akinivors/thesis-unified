"""
optimizer.py
============
Contextual Bandit Optimizer (CBO) decision engine.
Combines Guardrails, QTable, Estimator, and Reward tracking.
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
    Now supports both 'softmax' and 'epsilon_greedy' exploration.
    """
    
    def __init__(self, n_corpus: int, mode: str = "softmax"):
        self.n_corpus = n_corpus
        self.mode = mode
        
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
        self.n_min_visits = getattr(config, "N_MIN_VISITS", 10)
        self.n_warmup = getattr(config, "N_WARMUP", 2000)
        
        self.query_count = 0
        
        # Freeze State
        self.is_frozen = False
        self.frozen_crossover_point: Optional[float] = None

    def route(self, selectivity: float) -> str:
        """
        Returns strategy_name based on selectivity.
        If frozen, uses the learned crossover point directly.
        """
        if self.is_frozen and self.frozen_crossover_point is not None:
            arm = "bitmap_prefilter" if selectivity < self.frozen_crossover_point else "post_filter"
            return arm

        if selectivity < self.sigma_lower:
            return "bitmap_prefilter"
            
        if selectivity > self.sigma_upper:
            return "post_filter"
            
        n_visits = self.qtable.get_visits(selectivity)
        if n_visits < self.n_min_visits:
            # Random exploration until min visits
            arm = "bitmap_prefilter" if random.random() < 0.5 else "post_filter"
            return arm
            
        q_vals = self.qtable.get_q_values(selectivity)
        
        if self.mode == "softmax":
            arm = self._softmax_choice(q_vals)
        else:
            arm = self._epsilon_greedy_choice(q_vals)
            
        return arm

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
        Computes reward and updates Q-table.
        Checks for freeze condition after update.
        """
        reward = self.reward_fn.compute(latency_ms, recall)
        
        # Don't update if frozen or outside guardrails
        if not self.is_frozen and self.sigma_lower <= selectivity <= self.sigma_upper:
            self.qtable.update(selectivity, strategy_name, reward)
            
            # Check freeze condition dynamically
            crossover = self.qtable.check_freeze_condition()
            if crossover is not None:
                self.is_frozen = True
                self.frozen_crossover_point = crossover
            
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


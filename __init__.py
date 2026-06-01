"""
cbo — Contextual Bandit-based Cost-Based Optimizer
====================================================
A reinforcement-learning optimizer that routes filtered vector search
queries to the best execution strategy (bitmap pre-filter vs. post-filter)
based on learned Q-values indexed by filter selectivity.
"""

from cbo.optimizer import ContextualBanditOptimizer
from cbo.reward import SoftCliffReward
from cbo.guardrails import Guardrails
from cbo.qtable import QTable
from cbo.exploration import TierBased, ExponentialDecay, Softmax, ExplorationStrategy
from cbo.metrics import MetricsTracker, QueryRecord

__all__ = [
    "ContextualBanditOptimizer",
    "SoftCliffReward",
    "Guardrails",
    "QTable",
    "TierBased",
    "ExponentialDecay",
    "Softmax",
    "ExplorationStrategy",
    "MetricsTracker",
    "QueryRecord",
]

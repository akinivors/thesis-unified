"""
cbo.reward — Soft Cliff SLA Reward Function
=============================================
Combines latency and recall into a single scalar reward signal.

When recall meets the SLA target, reward is purely a function of speed.
Below the target, a steep power-law penalty (controlled by β) rapidly
drives reward toward zero, discouraging strategies that sacrifice recall.
"""

import logging

import config

log = logging.getLogger(__name__)


class SoftCliffReward:
    """Soft Cliff SLA reward function for the contextual bandit.

    The reward balances two objectives:
      • **Speed**: Normalized latency score ``L_norm = max(0, 1 - latency / L_max)``
      • **Recall SLA**: A steep cliff penalty when recall drops below ``r_target``

    Parameters
    ----------
    r_target : float
        Minimum acceptable recall (SLA target).
    l_max : float
        Maximum tolerable latency in milliseconds.  Anything at or above
        this value yields ``L_norm = 0``.
    beta : float
        Exponent controlling the steepness of the recall penalty cliff.
        Higher values create a sharper penalty boundary.
    """

    def __init__(
        self,
        r_target: float = config.CBO_R_TARGET,
        l_max: float = config.CBO_L_MAX,
        beta: float = config.CBO_BETA,
    ) -> None:
        if r_target <= 0.0 or r_target > 1.0:
            raise ValueError(f"r_target must be in (0, 1], got {r_target}")
        if l_max <= 0.0:
            raise ValueError(f"l_max must be positive, got {l_max}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")

        self.r_target = r_target
        self.l_max = l_max
        self.beta = beta

    def compute(self, latency_ms: float, recall: float) -> float:
        """Compute the Soft Cliff SLA reward.

        Parameters
        ----------
        latency_ms : float
            Observed query latency in milliseconds.
        recall : float
            Observed recall@K value in [0, 1].

        Returns
        -------
        float
            Reward in [0, 1].  Higher is better.

        Formula
        -------
        ``L_norm = max(0, 1 - latency_ms / L_max)``

        If ``recall >= r_target``::

            reward = L_norm          (pure speed reward)

        If ``recall < r_target``::

            reward = L_norm * (recall / r_target) ** beta
        """
        l_norm = max(0.0, 1.0 - latency_ms / self.l_max)

        if recall >= self.r_target:
            return l_norm

        # Guard against division by zero (r_target validated in __init__)
        ratio = recall / self.r_target
        return l_norm * ratio ** self.beta

    def __repr__(self) -> str:
        return (
            f"SoftCliffReward(r_target={self.r_target}, "
            f"l_max={self.l_max}, beta={self.beta})"
        )

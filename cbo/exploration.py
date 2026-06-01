"""
cbo.exploration — Exploration Strategies for the Contextual Bandit
===================================================================
Three complementary exploration strategies control how the optimizer
balances exploitation of the current best strategy against exploration
of the alternative:

  • **TierBased** — Discrete confidence tiers with hand-tuned ε values.
  • **ExponentialDecay** — Smooth ε = ε_max · e^(−k·Δ) based on Q-gap.
  • **Softmax** — Boltzmann selection with temperature τ.
"""

import logging
import math
import random
from abc import ABC, abstractmethod

import config

log = logging.getLogger(__name__)


class ExplorationStrategy(ABC):
    """Abstract base class for bandit exploration strategies.

    Subclasses must implement :meth:`choose`, which selects between
    ``"bitmap_prefilter"`` and ``"post_filter"`` given the current
    Q-value estimates for both strategies.
    """

    name: str = "base"

    @abstractmethod
    def choose(self, q_bitmap: float, q_post: float, step: int = 0) -> str:
        """Choose a strategy given current Q-values.

        Parameters
        ----------
        q_bitmap : float
            Current Q-value for ``bitmap_prefilter``.
        q_post : float
            Current Q-value for ``post_filter``.
        step : int, optional
            Current global step for decay logic.

        Returns
        -------
        str
            ``"bitmap_prefilter"`` or ``"post_filter"``.
        """

    def _exploit(self, q_bitmap: float, q_post: float) -> str:
        """Return the strategy with the higher Q-value (greedy)."""
        return "bitmap_prefilter" if q_bitmap >= q_post else "post_filter"

    def _explore(self, q_bitmap: float, q_post: float) -> str:
        """Return the strategy with the *lower* Q-value (forced exploration)."""
        return "post_filter" if q_bitmap >= q_post else "bitmap_prefilter"


class TierBased(ExplorationStrategy):
    """Discrete confidence-tier ε-greedy exploration.

    The exploration probability is selected from three tiers based on the
    absolute Q-value gap |Q_bitmap − Q_post|:

      +-----------------------+--------+
      | Tier                  |   ε    |
      +=======================+========+
      | High confidence Δ>0.3 |  0.05  |
      +-----------------------+--------+
      | Moderate  0.1<Δ≤0.3   |  0.15  |
      +-----------------------+--------+
      | Battleground  Δ≤0.1   |  0.40  |
      +-----------------------+--------+
    """

    name = "tier_based"

    # Tier thresholds and their exploration rates
    _HIGH_CONFIDENCE_THRESHOLD = 0.3
    _MODERATE_THRESHOLD = 0.1
    _EPSILON_HIGH = 0.05
    _EPSILON_MODERATE = 0.15
    _EPSILON_BATTLEGROUND = 0.40

    def choose(self, q_bitmap: float, q_post: float, step: int = 0) -> str:
        """Choose a strategy using tier-based ε-greedy with time decay."""
        delta = abs(q_bitmap - q_post)

        if delta > self._HIGH_CONFIDENCE_THRESHOLD:
            base_eps = self._EPSILON_HIGH
        elif delta > self._MODERATE_THRESHOLD:
            base_eps = self._EPSILON_MODERATE
        else:
            base_eps = self._EPSILON_BATTLEGROUND

        # Decay epsilon over time
        decay_factor = math.exp(-0.0005 * step)
        epsilon = max(0.01, base_eps * decay_factor)

        if random.random() < epsilon:
            chosen = self._explore(q_bitmap, q_post)
            log.debug(
                "TierBased EXPLORE: Δ=%.4f ε=%.2f → %s", delta, epsilon, chosen
            )
            return chosen

        return self._exploit(q_bitmap, q_post)


class ExponentialDecay(ExplorationStrategy):
    """Smooth exponential-decay ε-greedy exploration.

    ``ε = ε_max · e^(−k · Δ)``

    where Δ = |Q_bitmap − Q_post|.  As confidence grows (larger Δ),
    exploration decays exponentially.

    Parameters
    ----------
    epsilon_max : float
        Maximum exploration rate when Δ = 0.
    k : float
        Decay constant controlling how fast ε drops with increasing Δ.
    """

    name = "exponential_decay"

    def __init__(
        self,
        epsilon_max: float = config.CBO_EPSILON_MAX,
        k: float = config.CBO_DECAY_K,
    ) -> None:
        if epsilon_max < 0.0 or epsilon_max > 1.0:
            raise ValueError(f"epsilon_max must be in [0, 1], got {epsilon_max}")
        if k < 0.0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.epsilon_max = epsilon_max
        self.k = k

    def choose(self, q_bitmap: float, q_post: float, step: int = 0) -> str:
        """Choose a strategy using exponential-decay ε-greedy."""
        delta = abs(q_bitmap - q_post)
        curr_eps_max = max(0.01, self.epsilon_max * math.exp(-0.0005 * step))
        epsilon = curr_eps_max * math.exp(-self.k * delta)

        if random.random() < epsilon:
            chosen = self._explore(q_bitmap, q_post)
            log.debug(
                "ExponentialDecay EXPLORE: Δ=%.4f ε=%.4f → %s",
                delta,
                epsilon,
                chosen,
            )
            return chosen

        return self._exploit(q_bitmap, q_post)


class Softmax(ExplorationStrategy):
    """Boltzmann (softmax) exploration.

    Selects each strategy with probability proportional to
    ``e^(Q / τ)``, where τ is the temperature parameter.

    Lower τ → more greedy; higher τ → more uniform exploration.

    Uses the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    tau : float
        Temperature parameter.  Must be positive.
    """

    name = "softmax"

    def __init__(self, tau: float = config.CBO_TAU_INIT) -> None:
        if tau <= 0.0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau

    def choose(self, q_bitmap: float, q_post: float, step: int = 0) -> str:
        """Choose a strategy using Boltzmann selection with temperature decay."""
        # Numerically-stable softmax (subtract max before exp)
        max_q = max(q_bitmap, q_post)
        
        # Decay tau over time (Simulated Annealing)
        current_tau = max(0.01, self.tau * math.exp(-0.0005 * step))
        
        exp_bitmap = math.exp((q_bitmap - max_q) / current_tau)
        exp_post = math.exp((q_post - max_q) / current_tau)

        p_bitmap = exp_bitmap / (exp_bitmap + exp_post)

        chosen = "bitmap_prefilter" if random.random() < p_bitmap else "post_filter"
        log.debug(
            "Softmax: P(bitmap)=%.4f τ=%.4f → %s", p_bitmap, current_tau, chosen
        )
        return chosen

"""
cbo.guardrails — Stage 1 Hard Boundary Guard-rails
====================================================
For extreme selectivity values the optimal strategy is deterministic:

  • Very low selectivity  (σ < σ_lower):  bitmap pre-filter is mandatory
    because post-filter would discard almost every candidate, causing
    recall collapse.
  • Very high selectivity (σ > σ_upper):  post-filter is mandatory
    because a bitmap scan touching nearly every row wastes CPU without
    benefit.

The guardrail check runs *before* the bandit so that the exploration
engine never wastes budget on foregone-conclusion regions.
"""

import logging
from typing import Optional

import config

log = logging.getLogger(__name__)


class Guardrails:
    """Hard boundary guardrails that override the bandit at extremes.

    Parameters
    ----------
    sigma_lower : float
        Selectivity threshold below which ``bitmap_prefilter`` is forced.
    sigma_upper : float
        Selectivity threshold above which ``post_filter`` is forced.
    """

    def __init__(
        self,
        sigma_lower: float = config.CBO_SIGMA_LOWER,
        sigma_upper: float = config.CBO_SIGMA_UPPER,
    ) -> None:
        if not (0.0 <= sigma_lower < sigma_upper <= 1.0):
            raise ValueError(
                f"Need 0 <= sigma_lower < sigma_upper <= 1, "
                f"got sigma_lower={sigma_lower}, sigma_upper={sigma_upper}"
            )
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper

    def check(self, selectivity: float) -> Optional[str]:
        """Check whether a guardrail overrides the bandit decision.

        Parameters
        ----------
        selectivity : float
            Filter selectivity in [0, 1].

        Returns
        -------
        str or None
            ``"bitmap_prefilter"`` if selectivity is below the lower
            threshold, ``"post_filter"`` if above the upper threshold,
            or ``None`` if the bandit should decide.
        """
        if selectivity < self.sigma_lower:
            log.debug(
                "Guardrail: σ=%.4f < σ_lower=%.4f → bitmap_prefilter",
                selectivity,
                self.sigma_lower,
            )
            return "bitmap_prefilter"

        if selectivity > self.sigma_upper:
            log.debug(
                "Guardrail: σ=%.4f > σ_upper=%.4f → post_filter",
                selectivity,
                self.sigma_upper,
            )
            return "post_filter"

        return None

    def __repr__(self) -> str:
        return (
            f"Guardrails(sigma_lower={self.sigma_lower}, "
            f"sigma_upper={self.sigma_upper})"
        )

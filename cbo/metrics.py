"""
cbo.metrics — Telemetry and Analytics
======================================
Tracks every query decision made by the optimizer and provides
analytical views (cumulative regret, crossover convergence) plus
export to CSV and JSON for downstream analysis and plotting.
"""

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """Single query execution record capturing all observable quantities.

    Attributes
    ----------
    epoch : int
        Training epoch index (0-based).
    query_idx : int
        Index of the query within its epoch.
    global_step : int
        Monotonically increasing counter across all epochs.
    selectivity : float
        Filter selectivity for this query.
    strategy_chosen : str
        Strategy selected by the optimizer.
    latency_ms : float
        Observed latency in milliseconds.
    recall : float
        Observed recall@K.
    reward : float
        Computed reward from the Soft Cliff SLA function.
    q_bitmap : float
        Q-value for ``bitmap_prefilter`` *before* this update.
    q_post : float
        Q-value for ``post_filter`` *before* this update.
    crossover_estimate : float or None
        Current crossover estimate from the Q-table, if available.
    decision_overhead_us : float
        Time spent in the routing decision, in microseconds.
    was_guardrail : bool
        Whether the strategy was forced by a guardrail.
    oracle_strategy : str
        Ground-truth best strategy for this query.
    oracle_reward : float
        Reward that the oracle (best) strategy would have achieved.
    """

    epoch: int
    query_idx: int
    global_step: int
    selectivity: float
    strategy_chosen: str
    latency_ms: float
    recall: float
    reward: float
    q_bitmap: float
    q_post: float
    crossover_estimate: Optional[float]
    decision_overhead_us: float
    was_guardrail: bool
    oracle_strategy: str
    oracle_reward: float


class MetricsTracker:
    """Accumulates :class:`QueryRecord` instances and provides analytics.

    Examples
    --------
    >>> tracker = MetricsTracker()
    >>> rec = tracker.record(
    ...     epoch=0, query_idx=0, selectivity=0.25,
    ...     strategy_chosen="bitmap_prefilter", latency_ms=3.5,
    ...     recall=0.95, reward=0.825, q_bitmap=1.0, q_post=1.0,
    ...     crossover_estimate=0.25, decision_overhead_us=12.3,
    ...     was_guardrail=False, oracle_strategy="bitmap_prefilter",
    ...     oracle_reward=0.825,
    ... )
    >>> len(tracker.records)
    1
    """

    def __init__(self) -> None:
        self.records: List[QueryRecord] = []
        self._step_counter: int = 0

    def record(self, **kwargs) -> QueryRecord:
        """Create and store a new :class:`QueryRecord`.

        The ``global_step`` field is set automatically from an internal
        counter and should **not** be passed in *kwargs*.

        Parameters
        ----------
        **kwargs
            All fields of :class:`QueryRecord` except ``global_step``.

        Returns
        -------
        QueryRecord
            The newly created record.
        """
        rec = QueryRecord(global_step=self._step_counter, **kwargs)
        self.records.append(rec)
        self._step_counter += 1
        return rec

    # ── Analytics ────────────────────────────────────────────────────────

    def get_cumulative_regret(self) -> List[float]:
        """Compute cumulative regret over all recorded steps.

        Regret at step *t* is defined as::

            regret_t = max(0, oracle_reward_t − reward_t)
            cumulative_regret_T = Σ_{t=0}^{T} regret_t

        Returns
        -------
        list of float
            Cumulative regret at each step.
        """
        regret: List[float] = []
        total = 0.0
        for rec in self.records:
            total += max(0.0, rec.oracle_reward - rec.reward)
            regret.append(total)
        return regret

    def get_crossover_convergence(self) -> List[Optional[float]]:
        """Extract the crossover estimate at each step.

        Returns
        -------
        list of float or None
            Crossover estimate at each recorded step.
        """
        return [rec.crossover_estimate for rec in self.records]

    def get_accuracy(self) -> float:
        """Fraction of steps where the optimizer chose the oracle strategy.

        Returns
        -------
        float
            Accuracy in [0, 1], or 0.0 if no records exist.
        """
        if not self.records:
            return 0.0
        correct = sum(
            1 for rec in self.records
            if rec.strategy_chosen == rec.oracle_strategy
        )
        return correct / len(self.records)

    # ── Export ────────────────────────────────────────────────────────────

    def export_csv(self, path: Path) -> None:
        """Export all records to a CSV file.

        Parameters
        ----------
        path : Path
            Output file path.  Parent directories are created if needed.
        """
        if not self.records:
            log.warning("No records to export.")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(asdict(self.records[0]).keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))

        log.info("Exported %d records to %s", len(self.records), path)

    def export_json(self, path: Path) -> None:
        """Export all records to a JSON file.

        Parameters
        ----------
        path : Path
            Output file path.  Parent directories are created if needed.
        """
        if not self.records:
            log.warning("No records to export.")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

        log.info("Exported %d records to %s", len(self.records), path)

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"MetricsTracker(records={len(self.records)})"

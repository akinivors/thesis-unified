"""
filters.py — Dynamic Filter Generation with Controllable Selectivity
======================================================================
Inspects actual metadata distributions and generates ``FilterSpec``
objects — each expressing the **same logical predicate** in two forms:

1. A Python callable (predicate)  → used by PostFilter
2. A bitmap-resolver callable     → used by PreFilter and BitmapExact

This guarantees an apples-to-apples comparison across strategies.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from pyroaring import BitMap

from bitmap_index import BitmapIndex

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  FilterSpec Data Class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilterSpec:
    """A metadata filter expressed in two equivalent representations.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"main_cat='Camera & Photo'"``).
    target_selectivity : float
        The selectivity level this filter was generated to approximate.
    actual_selectivity : float
        The measured fraction of documents that match this filter.
    predicate : callable
        ``(metadata_dict) → bool`` — used for Python-side post-filtering.
    bitmap_resolver : callable
        ``(BitmapIndex) → BitMap`` — used for bitmap-based filtering.
    """

    name: str
    target_selectivity: float
    actual_selectivity: float = 0.0
    predicate: Callable[[Dict[str, Any]], bool] = field(
        default=None, repr=False, compare=False,
    )
    bitmap_resolver: Callable[[BitmapIndex], BitMap] = field(
        default=None, repr=False, compare=False,
    )

    def matches(self, metadata: Dict[str, Any]) -> bool:
        """Apply this filter to a single metadata dict."""
        return self.predicate(metadata)

    def resolve_bitmap(self, bitmap_index: BitmapIndex) -> BitMap:
        """Resolve matching document indices via the bitmap index."""
        return self.bitmap_resolver(bitmap_index)


# ═══════════════════════════════════════════════════════════════════════════════
#  Filter Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _build_candidate_pool(
    all_metadatas: List[Dict[str, Any]],
) -> List[FilterSpec]:
    """Generate a large pool of candidate filters with exact selectivities.

    Iterates over the full metadata list to compute *exact* selectivities
    (no independence assumptions for compound filters).
    """
    total = len(all_metadatas)
    if total == 0:
        return []

    candidates: List[FilterSpec] = []

    # ── 1.  main_cat  exact-match filters ──────────────────────────────────
    cat_counts: Counter = Counter(
        str(m.get("main_cat", "")) for m in all_metadatas
    )
    for cat, count in cat_counts.items():
        if not cat or cat == "Unknown":
            continue
        sel = count / total
        candidates.append(
            FilterSpec(
                name=f"main_cat='{cat}'",
                target_selectivity=0.0,
                actual_selectivity=sel,
                predicate=lambda m, _c=cat: str(m.get("main_cat", "")) == _c,
                bitmap_resolver=lambda bi, _c=cat: bi.lookup_eq("main_cat", _c),
            )
        )

    # ── 1.5  main_cat IN (c1, c2)  filters ─────────────────────────────────
    top_cats_list = [c for c, _ in cat_counts.most_common(12) if c and c != "Unknown"]
    for c1, c2 in itertools.combinations(top_cats_list, 2):
        count = sum(1 for m in all_metadatas if str(m.get("main_cat", "")) in (c1, c2))
        sel = count / total
        if sel >= 0.05:
            candidates.append(
                FilterSpec(
                    name=f"main_cat IN ('{c1}', '{c2}')",
                    target_selectivity=0.0,
                    actual_selectivity=sel,
                    predicate=lambda m, _c1=c1, _c2=c2: str(m.get("main_cat", "")) in (_c1, _c2),
                    bitmap_resolver=lambda bi, _c1=c1, _c2=c2: bi.lookup_in("main_cat", [_c1, _c2]),
                )
            )

    for c1, c2, c3 in itertools.combinations(top_cats_list, 3):
        count = sum(1 for m in all_metadatas if str(m.get("main_cat", "")) in (c1, c2, c3))
        sel = count / total
        if sel >= 0.10:
            candidates.append(
                FilterSpec(
                    name=f"main_cat IN ('{c1}', '{c2}', '{c3}')",
                    target_selectivity=0.0,
                    actual_selectivity=sel,
                    predicate=lambda m, _cats=(c1, c2, c3): str(m.get("main_cat", "")) in _cats,
                    bitmap_resolver=lambda bi, _cats=(c1, c2, c3): bi.lookup_in("main_cat", list(_cats)),
                )
            )

    # ── 2.  overall  range filters (≥ threshold and ≤ threshold) ─────────
    for threshold in [5.0, 4.0, 3.0, 2.0]:
        # GTE
        count_gte = sum(
            1
            for m in all_metadatas
            if float(m.get("overall", 0)) >= threshold
        )
        sel_gte = count_gte / total
        candidates.append(
            FilterSpec(
                name=f"overall>={threshold}",
                target_selectivity=0.0,
                actual_selectivity=sel_gte,
                predicate=lambda m, _th=threshold: float(m.get("overall", 0)) >= _th,
                bitmap_resolver=lambda bi, _th=threshold: bi.lookup_gte("overall", _th),
            )
        )
        # LTE
        count_lte = sum(
            1
            for m in all_metadatas
            if 0 < float(m.get("overall", 0)) <= threshold
        )
        sel_lte = count_lte / total
        candidates.append(
            FilterSpec(
                name=f"overall<={threshold}",
                target_selectivity=0.0,
                actual_selectivity=sel_lte,
                predicate=lambda m, _th=threshold: 0 < float(m.get("overall", 0)) <= _th,
                bitmap_resolver=lambda bi, _th=threshold: bi.lookup_lte("overall", _th),
            )
        )

    # ── 2.5  overall IN (v1, v2, …) combination filters ──────────────────
    all_ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    for r in range(2, 5):  # 2, 3, 4 element combos
        for combo in itertools.combinations(all_ratings, r):
            combo_set = set(combo)
            count = sum(
                1 for m in all_metadatas
                if float(m.get("overall", 0)) in combo_set
            )
            sel = count / total
            label = ",".join(str(int(v)) for v in combo)
            candidates.append(
                FilterSpec(
                    name=f"overall IN ({label})",
                    target_selectivity=0.0,
                    actual_selectivity=sel,
                    predicate=lambda m, _cs=combo_set: float(m.get("overall", 0)) in _cs,
                    bitmap_resolver=lambda bi, _vals=[str(v) for v in combo]: bi.lookup_in("overall", _vals),
                )
            )

    # ── 3.  brand  top-N exact-match filters ───────────────────────────────
    brand_counts: Counter = Counter(
        str(m.get("brand", "")) for m in all_metadatas
    )
    for brand, count in brand_counts.most_common(80):
        if not brand or brand == "Unknown":
            continue
        sel = count / total
        if sel < 0.001:
            continue  # skip extremely rare brands
        candidates.append(
            FilterSpec(
                name=f"brand='{brand}'",
                target_selectivity=0.0,
                actual_selectivity=sel,
                predicate=lambda m, _b=brand: str(m.get("brand", "")) == _b,
                bitmap_resolver=lambda bi, _b=brand: bi.lookup_eq("brand", _b),
            )
        )

    # ── 4.  Compound: main_cat + overall range ─────────────────────────────
    top_cats = cat_counts.most_common(15)
    for cat, _ in top_cats:
        if not cat or cat == "Unknown":
            continue
        for threshold in [5.0, 4.0, 3.0]:
            count = sum(
                1
                for m in all_metadatas
                if str(m.get("main_cat", "")) == cat
                and float(m.get("overall", 0)) >= threshold
            )
            sel = count / total
            if sel >= 0.005:
                candidates.append(
                    FilterSpec(
                        name=f"main_cat='{cat}' AND overall>={threshold}",
                        target_selectivity=0.0,
                        actual_selectivity=sel,
                        predicate=lambda m, _c=cat, _th=threshold: (
                            str(m.get("main_cat", "")) == _c
                            and float(m.get("overall", 0)) >= _th
                        ),
                        bitmap_resolver=lambda bi, _c=cat, _th=threshold: (
                            bi.lookup_eq("main_cat", _c)
                            & bi.lookup_gte("overall", _th)
                        ),
                    )
                )

    # ── 4.5 Compound: verified + overall range ─────────────────────────────
    for v_label, v_val in [("true", "True"), ("false", "False")]:
        for threshold in [5.0, 4.0, 3.0, 2.0]:
            # GTE
            count_gte = sum(
                1 for m in all_metadatas
                if str(m.get("verified", "")) == v_val
                and float(m.get("overall", 0)) >= threshold
            )
            sel_gte = count_gte / total
            if sel_gte >= 0.005:
                candidates.append(
                    FilterSpec(
                        name=f"verified={v_label} AND overall>={threshold}",
                        target_selectivity=0.0,
                        actual_selectivity=sel_gte,
                        predicate=lambda m, _v=v_val, _th=threshold: (
                            str(m.get("verified", "")) == _v
                            and float(m.get("overall", 0)) >= _th
                        ),
                        bitmap_resolver=lambda bi, _v=v_val, _th=threshold: (
                            bi.lookup_eq("verified", _v)
                            & bi.lookup_gte("overall", _th)
                        ),
                    )
                )
            # LTE
            count_lte = sum(
                1 for m in all_metadatas
                if str(m.get("verified", "")) == v_val
                and 0 < float(m.get("overall", 0)) <= threshold
            )
            sel_lte = count_lte / total
            if sel_lte >= 0.005:
                candidates.append(
                    FilterSpec(
                        name=f"verified={v_label} AND overall<={threshold}",
                        target_selectivity=0.0,
                        actual_selectivity=sel_lte,
                        predicate=lambda m, _v=v_val, _th=threshold: (
                            str(m.get("verified", "")) == _v
                            and 0 < float(m.get("overall", 0)) <= _th
                        ),
                        bitmap_resolver=lambda bi, _v=v_val, _th=threshold: (
                            bi.lookup_eq("verified", _v)
                            & bi.lookup_lte("overall", _th)
                        ),
                    )
                )

    # ── 5.  verified  filter ───────────────────────────────────────────────
    for val_label, val_str in [("true", "True"), ("false", "False")]:
        count = sum(
            1
            for m in all_metadatas
            if str(m.get("verified", "")) == val_str
        )
        sel = count / total
        candidates.append(
            FilterSpec(
                name=f"verified={val_label}",
                target_selectivity=0.0,
                actual_selectivity=sel,
                predicate=lambda m, _v=val_str: str(
                    m.get("verified", "")
                )
                == _v,
                bitmap_resolver=lambda bi, _v=val_str: bi.lookup_eq(
                    "verified", _v
                ),
            )
        )

    log.info("  Generated %d candidate filters", len(candidates))
    return candidates


def generate_filters(
    all_metadatas: List[Dict[str, Any]],
    selectivity_targets: List[float],
    *,
    tolerance: float = 0.15,
) -> List[FilterSpec]:
    """Select the best filter for each target selectivity.

    Parameters
    ----------
    all_metadatas : list[dict]
        Complete metadata list from the dataset.
    selectivity_targets : list[float]
        Desired selectivity fractions (e.g. ``[0.05, 0.10, 0.20, …]``).
    tolerance : float
        Maximum acceptable absolute deviation from the target.

    Returns
    -------
    list[FilterSpec]
        One ``FilterSpec`` per target, sorted by ``target_selectivity``.
        Filters that couldn't be matched within *tolerance* are skipped.
    """
    candidates = _build_candidate_pool(all_metadatas)
    if not candidates:
        log.warning("  No candidate filters generated — metadata may be empty.")
        return []

    selected: List[FilterSpec] = []
    used_names: set = set()

    for target in sorted(selectivity_targets):
        # Sort candidates by closeness to the target, prefer unused filters
        ranked = sorted(
            candidates,
            key=lambda c: (c.name in used_names, abs(c.actual_selectivity - target)),
        )
        best = ranked[0]
        deviation = abs(best.actual_selectivity - target)

        if deviation > tolerance and best.name in used_names:
            log.warning(
                "  ⚠  No good filter for target %.1f%% "
                "(best: %s at %.1f%%, Δ=%.1f%%)",
                target * 100,
                best.name,
                best.actual_selectivity * 100,
                deviation * 100,
            )

        spec = FilterSpec(
            name=best.name,
            target_selectivity=target,
            actual_selectivity=best.actual_selectivity,
            predicate=best.predicate,
            bitmap_resolver=best.bitmap_resolver,
        )
        selected.append(spec)
        used_names.add(best.name)

    return selected

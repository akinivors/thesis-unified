"""
bitmap_index.py — Roaring Bitmap Index for Microsecond Metadata Filtering
===========================================================================
Provides a lightweight, in-memory index that maps categorical and ordinal
metadata fields to Roaring Bitmaps.  Lookups execute in microseconds,
enabling the *Bitmap Filtering* strategy that decouples metadata filtering
from the vector-search engine entirely.

Persistence is handled via ``pickle`` (the heavy ``BitMap`` objects are
serialised with ``pyroaring``'s native ``serialize / deserialize``).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

from pyroaring import BitMap

import config

log = logging.getLogger(__name__)


class BitmapIndex:
    """Roaring Bitmap index over document metadata.

    For every (field, value) pair encountered during ingestion the index
    maintains a ``BitMap`` whose members are the *integer indices* of the
    documents carrying that value.  Two parallel dicts (``idx_to_id`` /
    ``id_to_idx``) map integer indices to string document-IDs
    and vice-versa.
    """

    def __init__(self) -> None:
        # {field_name: {value_string: BitMap}}
        self.field_bitmaps: Dict[str, Dict[str, BitMap]] = {}
        # integer index  ⇆  string document-ID
        self.idx_to_id: Dict[int, str] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.total_docs: int = 0

    # ── Construction ───────────────────────────────────────────────────────

    def add_document(
        self, idx: int, doc_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Register a single document in the bitmap index.

        Parameters
        ----------
        idx : int
            Sequential integer index (0-based) assigned during ingestion.
        doc_id : str
            The corresponding string document-ID.
        metadata : dict
            Flat metadata dict.
        """
        self.idx_to_id[idx] = doc_id
        self.id_to_idx[doc_id] = idx
        self.total_docs = max(self.total_docs, idx + 1)

        for field in config.BITMAP_FIELDS:
            value = metadata.get(field)
            if value is None:
                continue

            val_str = str(value)
            # Skip sentinel / missing values
            if not val_str or val_str == "-1.0":
                continue

            field_map = self.field_bitmaps.setdefault(field, {})
            bm = field_map.get(val_str)
            if bm is None:
                bm = BitMap()
                field_map[val_str] = bm
            bm.add(idx)

    # ── Query Primitives ───────────────────────────────────────────────────

    def lookup_eq(self, field: str, value: str) -> BitMap:
        """Exact match:  ``field == value``."""
        return self.field_bitmaps.get(field, {}).get(str(value), BitMap())

    def lookup_in(self, field: str, values: List[str]) -> BitMap:
        """Set membership:  ``field IN (v1, v2, …)``."""
        result = BitMap()
        field_map = self.field_bitmaps.get(field, {})
        for v in values:
            bm = field_map.get(str(v))
            if bm is not None:
                result |= bm
        return result

    def lookup_gte(self, field: str, threshold: float) -> BitMap:
        """Range query:  ``field >= threshold``  (numeric fields)."""
        result = BitMap()
        for val_str, bm in self.field_bitmaps.get(field, {}).items():
            try:
                if float(val_str) >= threshold:
                    result |= bm
            except ValueError:
                continue
        return result

    def lookup_lte(self, field: str, threshold: float) -> BitMap:
        """Range query:  ``field <= threshold``  (numeric fields)."""
        result = BitMap()
        for val_str, bm in self.field_bitmaps.get(field, {}).items():
            try:
                if float(val_str) <= threshold:
                    result |= bm
            except ValueError:
                continue
        return result

    # ── Introspection ──────────────────────────────────────────────────────

    def get_field_stats(self) -> Dict[str, Dict[str, int]]:
        """Return  ``{field: {value: document_count}}``  for every indexed field."""
        return {
            field: {val: len(bm) for val, bm in val_map.items()}
            for field, val_map in self.field_bitmaps.items()
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialise the entire index to *path* (pickle + pyroaring native)."""
        payload = {
            "field_bitmaps": {
                field: {val: bm.serialize() for val, bm in vmap.items()}
                for field, vmap in self.field_bitmaps.items()
            },
            "idx_to_id": self.idx_to_id,
            "id_to_idx": self.id_to_idx,
            "total_docs": self.total_docs,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / 1e6
        log.info("Bitmap index saved → %s  (%.1f MB)", path, size_mb)

    @classmethod
    def load(cls, path: Path) -> "BitmapIndex":
        """Deserialise an index previously saved with :meth:`save`."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        idx = cls()
        idx.idx_to_id = data["idx_to_id"]
        idx.id_to_idx = data["id_to_idx"]
        idx.total_docs = data["total_docs"]
        idx.field_bitmaps = {
            field: {
                val: BitMap.deserialize(ser) for val, ser in vmap.items()
            }
            for field, vmap in data["field_bitmaps"].items()
        }
        log.info(
            "Bitmap index loaded ← %s  (%d docs, %d fields)",
            path,
            idx.total_docs,
            len(idx.field_bitmaps),
        )
        return idx

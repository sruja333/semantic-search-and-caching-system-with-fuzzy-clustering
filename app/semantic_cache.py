from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from threading import RLock

import numpy as np


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    dominant_cluster: int
    result: dict
    access_count: int = 0
    seen_sequence: int = 0
    top_clusters: tuple[int, ...] = field(default_factory=tuple)


class SemanticCache:
    def __init__(self, similarity_threshold: float, max_entries: int) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: OrderedDict[int, CacheEntry] = OrderedDict()
        # Cluster partition index: cluster_id -> cache entry keys.
        # Lookup scans only this partition (or a few nearby partitions),
        # instead of scanning the whole cache.
        self._cluster_to_keys: dict[int, set[int]] = defaultdict(set)
        self._id_seq = 0
        self._request_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self._lock = RLock()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def lookup(
        self,
        query_embedding: np.ndarray,
        dominant_cluster: int,
        candidate_clusters: tuple[int, ...],
    ) -> tuple[CacheEntry | None, float]:
        with self._lock:
            self._request_count += 1
            candidate_keys: set[int] = set()
            # Cluster-aware probing prevents linear scans as cache grows.
            for cluster_id in candidate_clusters:
                candidate_keys.update(self._cluster_to_keys.get(cluster_id, set()))
            # Fallback to dominant partition if upstream candidate list is empty.
            if not candidate_keys and dominant_cluster is not None:
                candidate_keys.update(self._cluster_to_keys.get(dominant_cluster, set()))
            if not candidate_keys:
                self.miss_count += 1
                return None, 0.0

            best_key = None
            best_score = -1.0
            for key in candidate_keys:
                entry = self._entries.get(key)
                if entry is None:
                    continue
                score = self._cosine(query_embedding, entry.embedding)
                if score > best_score:
                    best_key = key
                    best_score = score

            if best_key is None or best_score < self.similarity_threshold:
                self.miss_count += 1
                return None, max(best_score, 0.0)

            entry = self._entries[best_key]
            entry.access_count += 1
            # LRU behavior: touched entries move to the back.
            self._entries.move_to_end(best_key, last=True)
            self.hit_count += 1
            return entry, best_score

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        dominant_cluster: int,
        top_clusters: tuple[int, ...],
        result: dict,
    ) -> None:
        with self._lock:
            self._id_seq += 1
            key = self._id_seq
            entry = CacheEntry(
                query=query,
                embedding=query_embedding.copy(),
                dominant_cluster=dominant_cluster,
                result=result,
                access_count=1,
                seen_sequence=key,
                top_clusters=top_clusters,
            )
            self._entries[key] = entry
            self._cluster_to_keys[dominant_cluster].add(key)
            for c in top_clusters:
                self._cluster_to_keys[c].add(key)

            while len(self._entries) > self.max_entries:
                old_key, old_entry = self._entries.popitem(last=False)
                self._cluster_to_keys[old_entry.dominant_cluster].discard(old_key)
                for c in old_entry.top_clusters:
                    self._cluster_to_keys[c].discard(old_key)

    def flush(self) -> None:
        with self._lock:
            self._entries.clear()
            self._cluster_to_keys.clear()
            self.hit_count = 0
            self.miss_count = 0
            self._request_count = 0

    def stats(self) -> dict:
        total = self.hit_count + self.miss_count
        return {
            "total_entries": len(self._entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": float(self.hit_count / total) if total else 0.0,
        }

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import minmax_scale

from .config import PROJECT_ROOT, SETTINGS
from .dataset import Document


@dataclass
class ClusterSelectionMetrics:
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    combined_score: float


class FuzzyClusterModel:
    def __init__(
        self,
        kmeans: MiniBatchKMeans,
        temperature: float,
        memberships: np.ndarray,
        labels: np.ndarray,
        selection_metrics: list[ClusterSelectionMetrics],
    ) -> None:
        self.kmeans = kmeans
        self.temperature = temperature
        self.memberships = memberships
        self.labels = labels
        self.selection_metrics = selection_metrics

    @property
    def n_clusters(self) -> int:
        return self.kmeans.n_clusters

    def membership_for_embedding(self, embedding: np.ndarray) -> np.ndarray:
        centers = self.kmeans.cluster_centers_.astype(np.float32)
        diff = centers - embedding.reshape(1, -1)
        distances = np.linalg.norm(diff, axis=1)
        probs = softmax(-(distances / self.temperature))
        return probs.astype(np.float32)

    def dominant_cluster(self, embedding: np.ndarray) -> int:
        return int(np.argmax(self.membership_for_embedding(embedding)))

    def save(self, path: Path | None = None) -> None:
        target = path or SETTINGS.clustering_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kmeans": self.kmeans,
                "temperature": self.temperature,
                "memberships": self.memberships,
                "labels": self.labels,
                "selection_metrics": [m.__dict__ for m in self.selection_metrics],
            },
            target,
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "FuzzyClusterModel":
        target = Path(path or SETTINGS.clustering_path)
        if not target.exists() and not target.is_absolute():
            fallback = PROJECT_ROOT / target
            if fallback.exists():
                target = fallback
        if not target.exists():
            raise FileNotFoundError(
                f"Clustering artifact not found at '{target}'. "
                f"Run `python scripts/build_index.py` to generate artifacts."
            )
        payload = joblib.load(target)
        metrics = [ClusterSelectionMetrics(**m) for m in payload["selection_metrics"]]
        return cls(
            kmeans=payload["kmeans"],
            temperature=float(payload["temperature"]),
            memberships=payload["memberships"],
            labels=payload["labels"],
            selection_metrics=metrics,
        )


def _combined_metric_score(metrics: list[ClusterSelectionMetrics]) -> list[float]:
    sil = np.array([m.silhouette for m in metrics], dtype=np.float64)
    ch = np.array([m.calinski_harabasz for m in metrics], dtype=np.float64)
    db = np.array([m.davies_bouldin for m in metrics], dtype=np.float64)
    sil_n = minmax_scale(sil)
    ch_n = minmax_scale(np.log1p(ch))
    db_n = minmax_scale(-db)
    combined = 0.5 * sil_n + 0.3 * ch_n + 0.2 * db_n
    return combined.tolist()


def select_cluster_count(
    embeddings: np.ndarray, candidates: tuple[int, ...] | None = None
) -> list[ClusterSelectionMetrics]:
    ks = candidates or SETTINGS.cluster_candidates
    raw_metrics: list[ClusterSelectionMetrics] = []

    for k in ks:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=SETTINGS.random_state,
            batch_size=SETTINGS.kmeans_batch_size,
            n_init=10,
        )
        labels = model.fit_predict(embeddings)
        sil = silhouette_score(
            embeddings,
            labels,
            metric="euclidean",
            sample_size=min(5000, len(embeddings)),
            random_state=SETTINGS.random_state,
        )
        ch = calinski_harabasz_score(embeddings, labels)
        db = davies_bouldin_score(embeddings, labels)
        raw_metrics.append(
            ClusterSelectionMetrics(
                k=k,
                silhouette=float(sil),
                calinski_harabasz=float(ch),
                davies_bouldin=float(db),
                combined_score=0.0,
            )
        )

    combined = _combined_metric_score(raw_metrics)
    for i, score in enumerate(combined):
        raw_metrics[i].combined_score = float(score)
    return raw_metrics


def _pairwise_distances_to_centers(
    embeddings: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    # Uses ||x-c||^2 = ||x||^2 + ||c||^2 - 2x.c for speed and memory safety.
    x_sq = np.sum(embeddings * embeddings, axis=1, keepdims=True)
    c_sq = np.sum(centers * centers, axis=1).reshape(1, -1)
    dot = embeddings @ centers.T
    sq = np.maximum(x_sq + c_sq - 2.0 * dot, 0.0)
    return np.sqrt(sq).astype(np.float32)


def fit_fuzzy_clusters(embeddings: np.ndarray) -> FuzzyClusterModel:
    metrics = select_cluster_count(embeddings)
    best = max(metrics, key=lambda m: m.combined_score)

    kmeans = MiniBatchKMeans(
        n_clusters=best.k,
        random_state=SETTINGS.random_state,
        batch_size=SETTINGS.kmeans_batch_size,
        n_init=20,
    )
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_.astype(np.float32)
    distances = _pairwise_distances_to_centers(embeddings.astype(np.float32), centers)

    temp = float(np.percentile(distances, 35))
    temp = max(temp, 1e-4)
    memberships = softmax(-(distances / temp), axis=1).astype(np.float32)

    return FuzzyClusterModel(
        kmeans=kmeans,
        temperature=temp,
        memberships=memberships,
        labels=labels.astype(np.int32),
        selection_metrics=metrics,
    )


def build_cluster_report(
    documents: list[Document],
    memberships: np.ndarray,
    labels: np.ndarray,
    selection_metrics: list[ClusterSelectionMetrics],
    out_path: Path | None = None,
) -> dict:
    target = out_path or SETTINGS.cluster_report_path
    target.parent.mkdir(parents=True, exist_ok=True)

    n_clusters = memberships.shape[1]
    sorted_membership_idx = np.argsort(-memberships, axis=1)
    top1_scores = memberships[np.arange(len(memberships)), sorted_membership_idx[:, 0]]
    top2_scores = memberships[np.arange(len(memberships)), sorted_membership_idx[:, 1]]
    ambiguity_gap = top1_scores - top2_scores
    entropy = -(memberships * np.log(memberships + 1e-12)).sum(axis=1)

    report: dict = {
        "selected_cluster_count": int(n_clusters),
        "selection_metrics": [m.__dict__ for m in selection_metrics],
        "global_uncertain_docs": [],
        "clusters": [],
    }

    global_uncertain_idx = np.argsort(-entropy)[:20]
    report["global_uncertain_docs"] = [
        {
            "doc_id": documents[int(i)].doc_id,
            "top_two_clusters": [
                int(sorted_membership_idx[int(i), 0]),
                int(sorted_membership_idx[int(i), 1]),
            ],
            "top_two_probs": [
                float(top1_scores[int(i)]),
                float(top2_scores[int(i)]),
            ],
            "top_two_gap": float(ambiguity_gap[int(i)]),
            "entropy": float(entropy[int(i)]),
            "dominant_cluster": int(labels[int(i)]),
            "preview": documents[int(i)].raw_preview[:180],
        }
        for i in global_uncertain_idx
    ]

    for cluster_id in range(n_clusters):
        membership_values = memberships[:, cluster_id]
        top_idx = np.argsort(-membership_values)[:10]
        cluster_boundary_mask = (
            (sorted_membership_idx[:, 0] == cluster_id)
            | (sorted_membership_idx[:, 1] == cluster_id)
        )
        cluster_boundary_candidates = np.where(cluster_boundary_mask)[0]
        if len(cluster_boundary_candidates) > 0:
            rel = np.argsort(-entropy[cluster_boundary_candidates])[:10]
            boundary_idx = cluster_boundary_candidates[rel]
        else:
            boundary_idx = np.array([], dtype=np.int32)

        label_counts: dict[str, int] = {}
        for idx in np.where(labels == cluster_id)[0]:
            lbl = documents[int(idx)].label
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        dominant_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        report["clusters"].append(
            {
                "cluster_id": cluster_id,
                "size": int((labels == cluster_id).sum()),
                "dominant_labels": dominant_labels,
                "representative_docs": [
                    {
                        "doc_id": documents[int(i)].doc_id,
                        "membership": float(membership_values[int(i)]),
                        "preview": documents[int(i)].raw_preview[:180],
                    }
                    for i in top_idx
                ],
                "boundary_docs": [
                    {
                        "doc_id": documents[int(i)].doc_id,
                        "membership": float(membership_values[int(i)]),
                        "top_two_clusters": [
                            int(sorted_membership_idx[int(i), 0]),
                            int(sorted_membership_idx[int(i), 1]),
                        ],
                        "top_two_probs": [
                            float(top1_scores[int(i)]),
                            float(top2_scores[int(i)]),
                        ],
                        "top_two_gap": float(ambiguity_gap[int(i)]),
                        "entropy": float(entropy[int(i)]),
                        "preview": documents[int(i)].raw_preview[:180],
                    }
                    for i in boundary_idx
                ],
            }
        )

    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

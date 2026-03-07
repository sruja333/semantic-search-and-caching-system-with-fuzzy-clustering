from __future__ import annotations

import json

import numpy as np
from sklearn.preprocessing import normalize

from .clustering import FuzzyClusterModel
from .config import SETTINGS
from .semantic_cache import SemanticCache
from .vector_store import LocalVectorDB


def _query_from_clean_text(clean_text: str, max_tokens: int = 18) -> str:
    tokens = clean_text.split()
    return " ".join(tokens[:max_tokens])


def _build_query_feature_map(
    vector_db: LocalVectorDB,
    cluster_model: FuzzyClusterModel,
    query_texts: list[str],
) -> dict[str, tuple[np.ndarray, int, tuple[int, ...]]]:
    sparse = vector_db.vectorizer.transform(query_texts)
    dense = vector_db.svd.transform(sparse)
    dense = normalize(dense, norm="l2", axis=1).astype(np.float32)
    features: dict[str, tuple[np.ndarray, int, tuple[int, ...]]] = {}
    for i, query_text in enumerate(query_texts):
        emb = dense[i]
        membership = cluster_model.membership_for_embedding(emb)
        dominant_cluster = int(membership.argmax())
        top_clusters = tuple(membership.argsort()[-3:][::-1].astype(int).tolist())
        features[query_text] = (emb, dominant_cluster, top_clusters)
    return features


def _run_cache_probe(
    cache: SemanticCache,
    query_features: dict[str, tuple[np.ndarray, int, tuple[int, ...]]],
    query_text: str,
) -> bool:
    emb, dominant_cluster, top_clusters = query_features[query_text]
    hit, _ = cache.lookup(
        query_embedding=emb,
        dominant_cluster=dominant_cluster,
        candidate_clusters=top_clusters,
    )
    if hit is None:
        cache.store(
            query=query_text,
            query_embedding=emb.copy(),
            dominant_cluster=dominant_cluster,
            top_clusters=top_clusters,
            result={"synthetic": True},
        )
        return False
    return True


def run_cache_threshold_study(
    vector_db: LocalVectorDB,
    cluster_model: FuzzyClusterModel,
) -> dict:
    rng = np.random.default_rng(SETTINGS.random_state)
    embeddings = vector_db.embeddings
    metadata = vector_db.metadata
    labels = np.array([row["label"] for row in metadata], dtype=object)

    usable_idx = np.array(
        [i for i, row in enumerate(metadata) if len(row["cleaned_text"].split()) >= 18],
        dtype=np.int32,
    )
    if len(usable_idx) < 120:
        raise RuntimeError("Not enough usable documents for threshold study.")

    rng.shuffle(usable_idx)
    anchor_idx = usable_idx[:120]

    label_to_idx: dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        label_to_idx[label] = np.where(labels == label)[0]

    scenarios: list[tuple[str, str, str]] = []
    for idx in anchor_idx:
        text_a = _query_from_clean_text(metadata[int(idx)]["cleaned_text"])
        scenarios.append(("exact_duplicate", text_a, text_a))

        same_pool = label_to_idx[str(labels[int(idx)])]
        same_pool = same_pool[same_pool != idx]
        if len(same_pool) > 0:
            sample_pool = rng.choice(same_pool, size=min(80, len(same_pool)), replace=False)
            sims = embeddings[sample_pool] @ embeddings[int(idx)]
            pair_idx = int(sample_pool[int(np.argmax(sims))])
            text_b = _query_from_clean_text(metadata[pair_idx]["cleaned_text"])
            scenarios.append(("same_topic_rephrase", text_a, text_b))

        diff_pool = np.where(labels != labels[int(idx)])[0]
        sample_pool = rng.choice(diff_pool, size=min(100, len(diff_pool)), replace=False)
        sims = embeddings[sample_pool] @ embeddings[int(idx)]
        pair_idx = int(sample_pool[int(np.argmin(sims))])
        text_c = _query_from_clean_text(metadata[pair_idx]["cleaned_text"])
        scenarios.append(("different_topic", text_a, text_c))

    unique_queries = sorted({q1 for _, q1, _ in scenarios} | {q2 for _, _, q2 in scenarios})
    query_features = _build_query_feature_map(
        vector_db=vector_db, cluster_model=cluster_model, query_texts=unique_queries
    )

    threshold_rows = []
    for threshold in SETTINGS.cache_threshold_candidates:
        cache = SemanticCache(similarity_threshold=threshold, max_entries=50_000)
        scenario_stats = {
            "exact_duplicate": {"total": 0, "hit": 0},
            "same_topic_rephrase": {"total": 0, "hit": 0},
            "different_topic": {"total": 0, "hit": 0},
        }
        for scenario, q1, q2 in scenarios:
            _run_cache_probe(cache, query_features, q1)
            second_hit = _run_cache_probe(cache, query_features, q2)
            scenario_stats[scenario]["total"] += 1
            if second_hit:
                scenario_stats[scenario]["hit"] += 1

        row = {
            "threshold": float(threshold),
            "exact_hit_rate": scenario_stats["exact_duplicate"]["hit"]
            / max(scenario_stats["exact_duplicate"]["total"], 1),
            "same_topic_hit_rate": scenario_stats["same_topic_rephrase"]["hit"]
            / max(scenario_stats["same_topic_rephrase"]["total"], 1),
            "different_topic_false_hit_rate": scenario_stats["different_topic"]["hit"]
            / max(scenario_stats["different_topic"]["total"], 1),
        }
        row["tradeoff_score"] = (
            0.65 * row["same_topic_hit_rate"]
            + 0.35 * row["exact_hit_rate"]
            - 1.0 * row["different_topic_false_hit_rate"]
        )
        threshold_rows.append(row)

    best = max(threshold_rows, key=lambda x: x["tradeoff_score"])
    report = {
        "method": "cluster-aware semantic cache simulation over exact/same-topic/different-topic query pairs",
        "rows": threshold_rows,
        "recommended_threshold": best["threshold"],
        "selection_reason": "max tradeoff_score (high same-topic reuse, low cross-topic false hits)",
    }
    SETTINGS.cache_threshold_report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

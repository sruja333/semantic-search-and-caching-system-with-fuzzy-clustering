from __future__ import annotations

import json

from .cache_threshold import run_cache_threshold_study
from .clustering import FuzzyClusterModel, build_cluster_report, fit_fuzzy_clusters
from .config import SETTINGS
from .dataset import load_documents
from .semantic_cache import SemanticCache
from .vector_store import LocalVectorDB, SearchResult


class SemanticSearchEngine:
    def __init__(
        self,
        vector_db: LocalVectorDB,
        cluster_model: FuzzyClusterModel,
        cache: SemanticCache,
    ) -> None:
        self.vector_db = vector_db
        self.cluster_model = cluster_model
        self.cache = cache

    def _format_result(self, results: list[SearchResult]) -> dict:
        return {
            "top_matches": [
                {
                    "doc_id": row.doc_id,
                    "label": row.label,
                    "score": round(row.score, 4),
                    "preview": row.preview,
                }
                for row in results
            ]
        }

    def query(self, query_text: str) -> dict:
        query_embedding = self.vector_db.encode_query(query_text)
        membership = self.cluster_model.membership_for_embedding(query_embedding)
        dominant_cluster = int(membership.argmax())
        # Probe a few highly probable clusters, not just one.
        top_clusters = tuple(membership.argsort()[-3:][::-1].astype(int).tolist())

        hit, similarity = self.cache.lookup(
            query_embedding=query_embedding,
            dominant_cluster=dominant_cluster,
            candidate_clusters=top_clusters,
        )
        if hit:
            return {
                "query": query_text,
                "cache_hit": True,
                "matched_query": hit.query,
                "similarity_score": round(float(similarity), 4),
                "result": hit.result,
                "dominant_cluster": dominant_cluster,
            }

        result_rows = self.vector_db.search(query_embedding, top_k=SETTINGS.search_top_k)
        result_payload = self._format_result(result_rows)
        self.cache.store(
            query=query_text,
            query_embedding=query_embedding,
            dominant_cluster=dominant_cluster,
            top_clusters=top_clusters,
            result=result_payload,
        )
        return {
            "query": query_text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": 0.0,
            "result": result_payload,
            "dominant_cluster": dominant_cluster,
        }


def build_artifacts() -> dict:
    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    documents = load_documents()

    vector_db = LocalVectorDB()
    embeddings, _tfidf_matrix = vector_db.fit(documents)
    vector_db.save(SETTINGS.vector_store_path)

    clusters = fit_fuzzy_clusters(embeddings)
    clusters.save(SETTINGS.clustering_path)

    cluster_report = build_cluster_report(
        documents=documents,
        memberships=clusters.memberships,
        labels=clusters.labels,
        selection_metrics=clusters.selection_metrics,
        out_path=SETTINGS.cluster_report_path,
    )
    cache_threshold_report = run_cache_threshold_study(
        vector_db=vector_db,
        cluster_model=clusters,
    )

    build_report = {
        "documents_indexed": len(documents),
        "vector_store_path": str(SETTINGS.vector_store_path),
        "clustering_path": str(SETTINGS.clustering_path),
        "cluster_report_path": str(SETTINGS.cluster_report_path),
        "cache_threshold_report_path": str(SETTINGS.cache_threshold_report_path),
        "selected_clusters": cluster_report["selected_cluster_count"],
        "cache_similarity_threshold": SETTINGS.cache_similarity_threshold,
        "cache_threshold_recommended": cache_threshold_report["recommended_threshold"],
    }
    SETTINGS.build_report_path.write_text(
        json.dumps(build_report, indent=2), encoding="utf-8"
    )
    return build_report


def load_engine() -> SemanticSearchEngine:
    if not SETTINGS.vector_store_path.exists() or not SETTINGS.clustering_path.exists():
        raise FileNotFoundError(
            "Artifacts are missing. Run `python scripts/build_index.py` first."
        )
    threshold = SETTINGS.cache_similarity_threshold
    if SETTINGS.cache_threshold_report_path.exists():
        try:
            payload = json.loads(
                SETTINGS.cache_threshold_report_path.read_text(encoding="utf-8")
            )
            threshold = float(payload.get("recommended_threshold", threshold))
        except (ValueError, OSError, json.JSONDecodeError):
            threshold = SETTINGS.cache_similarity_threshold

    vector_db = LocalVectorDB.load(SETTINGS.vector_store_path)
    cluster_model = FuzzyClusterModel.load(SETTINGS.clustering_path)
    cache = SemanticCache(
        similarity_threshold=threshold,
        max_entries=SETTINGS.cache_max_entries,
    )
    return SemanticSearchEngine(vector_db=vector_db, cluster_model=cluster_model, cache=cache)

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_root: Path = PROJECT_ROOT / "data" / "20_newsgroups"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    min_clean_tokens: int = 8
    tfidf_max_features: int = 80_000
    svd_components: int = 256
    random_state: int = 42

    # Candidate values evaluated to select a data-driven cluster count.
    cluster_candidates: tuple[int, ...] = (12, 16, 20, 24, 28, 32)
    kmeans_batch_size: int = 4096
    cache_similarity_threshold: float = 0.80
    cache_threshold_candidates: tuple[float, ...] = (
        0.80,
        0.85,
        0.90,
        0.95,
    )
    cache_max_entries: int = 5000
    search_top_k: int = 5

    vector_store_path: Path = PROJECT_ROOT / "artifacts" / "vector_store.joblib"
    clustering_path: Path = PROJECT_ROOT / "artifacts" / "fuzzy_clusters.joblib"
    cluster_report_path: Path = PROJECT_ROOT / "artifacts" / "cluster_report.json"
    cache_threshold_report_path: Path = (
        PROJECT_ROOT / "artifacts" / "cache_threshold_report.json"
    )
    build_report_path: Path = PROJECT_ROOT / "artifacts" / "build_report.json"


SETTINGS = Settings()

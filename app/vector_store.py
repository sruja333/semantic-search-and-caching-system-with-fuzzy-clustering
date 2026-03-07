from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .config import PROJECT_ROOT, SETTINGS
from .dataset import Document
from .text_cleaning import clean_text

USENET_NOISE_STOPWORDS = {
    "article",
    "believe",
    "did",
    "does",
    "don",
    "drive",
    "good",
    "just",
    "know",
    "like",
    "make",
    "made",
    "people",
    "question",
    "questions",
    "read",
    "really",
    "said",
    "say",
    "says",
    "think",
    "thanks",
    "time",
    "used",
    "using",
    "use",
    "want",
    "way",
    "writes",
    "year",
    "years",
}


@dataclass
class SearchResult:
    doc_id: str
    label: str
    score: float
    preview: str


HEADER_NOISE_TOKENS = (
    "xref:",
    "path:",
    "newsgroups:",
    "from:",
    "organization:",
    "nntp-posting-host:",
    "distribution:",
    "lines:",
)


def _format_preview(row: dict, max_chars: int = 260) -> str:
    raw = str(row.get("preview", "")).strip()
    cleaned = str(row.get("cleaned_text", "")).strip()
    raw_l = raw.lower()

    # Many 20NG docs start with transport headers; prefer cleaned body text in that case.
    header_hits = sum(token in raw_l for token in HEADER_NOISE_TOKENS)
    looks_like_header_blob = header_hits >= 2 or raw.count("!") >= 3
    chosen = cleaned if (looks_like_header_blob and cleaned) else (raw or cleaned)
    chosen = re.sub(r"\s+", " ", chosen).strip()
    return chosen[:max_chars]


class LocalVectorDB:
    def __init__(
        self,
        vectorizer: TfidfVectorizer | None = None,
        svd: TruncatedSVD | None = None,
        nn_index: NearestNeighbors | None = None,
        embeddings: np.ndarray | None = None,
        metadata: list[dict] | None = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.svd = svd
        self.nn_index = nn_index
        self.embeddings = embeddings
        self.metadata = metadata or []

    @property
    def is_fitted(self) -> bool:
        return (
            self.vectorizer is not None
            and self.svd is not None
            and self.nn_index is not None
            and self.embeddings is not None
        )

    def fit(self, documents: list[Document]) -> tuple[np.ndarray, np.ndarray]:
        texts = [doc.cleaned_text for doc in documents]
        stop_words = sorted(ENGLISH_STOP_WORDS.union(USENET_NOISE_STOPWORDS))
        # Word + bigram TF-IDF keeps signal from short topical phrases (e.g., "space shuttle").
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.55,
            min_df=5,
            max_features=SETTINGS.tfidf_max_features,
            stop_words=stop_words,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # SVD converts sparse lexical vectors into dense semantic directions for fast ANN lookup.
        self.svd = TruncatedSVD(
            n_components=SETTINGS.svd_components,
            random_state=SETTINGS.random_state,
        )
        dense = self.svd.fit_transform(tfidf_matrix)
        dense = normalize(dense, norm="l2", axis=1).astype(np.float32)

        self.nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_index.fit(dense)
        self.embeddings = dense
        self.metadata = [
            {
                "doc_id": doc.doc_id,
                "label": doc.label,
                "preview": doc.raw_preview,
                "cleaned_text": doc.cleaned_text,
            }
            for doc in documents
        ]
        return dense, tfidf_matrix

    def encode_query(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Vector DB is not fitted.")
        cleaned = clean_text(query)
        vec = self.vectorizer.transform([cleaned if cleaned else query.lower()])
        dense = self.svd.transform(vec)
        dense = normalize(dense, norm="l2", axis=1).astype(np.float32)
        return dense[0]

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if not self.is_fitted:
            raise RuntimeError("Vector DB is not fitted.")
        query_2d = query_embedding.reshape(1, -1)
        distances, indices = self.nn_index.kneighbors(
            query_2d, n_neighbors=min(top_k, len(self.metadata))
        )
        out: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            row = self.metadata[idx]
            out.append(
                SearchResult(
                    doc_id=row["doc_id"],
                    label=row["label"],
                    score=float(1.0 - dist),
                    preview=_format_preview(row),
                )
            )
        return out

    def save(self, path: Path | None = None) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted vector DB.")
        target = path or SETTINGS.vector_store_path
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "nn_index": self.nn_index,
                "embeddings": self.embeddings,
                "metadata": self.metadata,
            },
            target,
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "LocalVectorDB":
        target = Path(path or SETTINGS.vector_store_path)
        if not target.exists() and not target.is_absolute():
            fallback = PROJECT_ROOT / target
            if fallback.exists():
                target = fallback
        if not target.exists():
            raise FileNotFoundError(
                f"Vector store not found at '{target}'. "
                f"Run `python scripts/build_index.py` to generate artifacts."
            )
        payload = joblib.load(target)
        return cls(
            vectorizer=payload["vectorizer"],
            svd=payload["svd"],
            nn_index=payload["nn_index"],
            embeddings=payload["embeddings"],
            metadata=payload["metadata"],
        )

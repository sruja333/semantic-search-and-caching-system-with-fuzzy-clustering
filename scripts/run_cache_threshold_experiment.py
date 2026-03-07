import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vector_store import LocalVectorDB


if __name__ == "__main__":
    vector_db = LocalVectorDB.load()

    # Reproducible, fast sample for threshold behavior analysis.
    sample_size = 500
    query_embeddings = vector_db.embeddings[:sample_size]
    thresholds = [0.80, 0.85, 0.90, 0.95]

    results: list[dict] = []

    for threshold in thresholds:
        cache: list[np.ndarray] = []
        hits = 0
        similarities: list[float] = []

        for emb in query_embeddings:
            if not cache:
                cache.append(emb)
                continue

            sims = np.dot(np.vstack(cache), emb)
            best = float(sims.max())

            if best >= threshold:
                hits += 1
                similarities.append(best)
            else:
                cache.append(emb)

        hit_rate = hits / len(query_embeddings)
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        results.append(
            {
                "threshold": threshold,
                "hit_rate": hit_rate,
                "avg_similarity": avg_similarity,
            }
        )

    report = {"sample_size": sample_size, "results": results}
    out_path = ROOT / "artifacts" / "cache_threshold_experiment.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    thresholds_plot = [row["threshold"] for row in results]
    hit_rates_plot = [row["hit_rate"] for row in results]
    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds_plot, hit_rates_plot, marker="o")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Cache Hit Rate")
    plt.title("Semantic Cache Threshold Tradeoff")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = ROOT / "artifacts" / "cache_threshold_tradeoff.png"
    plt.savefig(plot_path, dpi=140)
    plt.close()

    print("\nCache Threshold Experiment\n")
    print("Threshold | Hit Rate | Avg Similarity")
    print("--------------------------------------")
    for row in results:
        print(
            f"{row['threshold']:.2f}      | "
            f"{row['hit_rate'] * 100:5.1f}%   | "
            f"{row['avg_similarity']:.3f}"
        )
    print(f"\nSaved plot: {plot_path}")

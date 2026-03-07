from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clustering import FuzzyClusterModel
from app.config import SETTINGS
from app.vector_store import LocalVectorDB


def main(top_n: int = 10) -> None:
    vector_db = LocalVectorDB.load(SETTINGS.vector_store_path)
    cluster_model = FuzzyClusterModel.load(SETTINGS.clustering_path)

    memberships = cluster_model.memberships
    labels = cluster_model.labels
    metadata = vector_db.metadata

    entropy = -(memberships * np.log(memberships + 1e-9)).sum(axis=1)
    top_idx = np.argsort(-entropy)[:top_n]

    out = []
    for i in top_idx:
        sorted_idx = np.argsort(-memberships[i])[:2]
        p1 = float(memberships[i, sorted_idx[0]])
        p2 = float(memberships[i, sorted_idx[1]])
        out.append(
            {
                "doc_id": metadata[int(i)]["doc_id"],
                "label": metadata[int(i)]["label"],
                "dominant_cluster": int(labels[int(i)]),
                "top_two_clusters": [int(sorted_idx[0]), int(sorted_idx[1])],
                "top_two_probs": [p1, p2],
                "top_two_gap": float(p1 - p2),
                "entropy": float(entropy[int(i)]),
                "preview": metadata[int(i)]["preview"][:220],
            }
        )

    out_path = ROOT / "artifacts" / "boundary_documents.json"
    out_path.write_text(json.dumps({"top_n": top_n, "documents": out}, indent=2), encoding="utf-8")

    print("\nTop Entropy Boundary Documents\n")
    for row in out:
        print(
            f"{row['doc_id']} | C{row['top_two_clusters'][0]}:{row['top_two_probs'][0]:.4f} "
            f"vs C{row['top_two_clusters'][1]}:{row['top_two_probs'][1]:.4f} | "
            f"entropy={row['entropy']:.4f}"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main(top_n=10)


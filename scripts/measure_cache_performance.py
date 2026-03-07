from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine import load_engine


def _to_query(text: str, max_tokens: int = 14) -> str:
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def _p95_ms(latencies_ms: list[float]) -> float:
    if not latencies_ms:
        return 0.0
    ordered = sorted(latencies_ms)
    idx = int(0.95 * (len(ordered) - 1))
    return float(ordered[idx])


def main() -> None:
    engine = load_engine()

    rng = random.Random(42)
    all_queries = [_to_query(row["cleaned_text"]) for row in engine.vector_db.metadata]
    all_queries = [q for q in all_queries if len(q.split()) >= 6]
    rng.shuffle(all_queries)
    queries = all_queries[:250]

    # Miss benchmark: one pass over unseen queries in an empty cache.
    engine.cache.flush()
    miss_latencies_ms: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        response = engine.query(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not response["cache_hit"]:
            miss_latencies_ms.append(elapsed_ms)

    # Hit benchmark: insert then immediately query same text again.
    engine.cache.flush()
    hit_latencies_ms: list[float] = []
    for q in queries:
        engine.query(q)  # populate cache
        t0 = time.perf_counter()
        response = engine.query(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if response["cache_hit"]:
            hit_latencies_ms.append(elapsed_ms)

    miss_mean = float(statistics.mean(miss_latencies_ms)) if miss_latencies_ms else 0.0
    hit_mean = float(statistics.mean(hit_latencies_ms)) if hit_latencies_ms else 0.0
    speedup = (miss_mean / hit_mean) if hit_mean > 0 else 0.0

    payload = {
        "queries_tested": len(queries),
        "hit_measurements": len(hit_latencies_ms),
        "miss_measurements": len(miss_latencies_ms),
        "hit_latency_ms": {
            "mean": hit_mean,
            "median": float(statistics.median(hit_latencies_ms)) if hit_latencies_ms else 0.0,
            "p95": _p95_ms(hit_latencies_ms),
        },
        "miss_latency_ms": {
            "mean": miss_mean,
            "median": float(statistics.median(miss_latencies_ms)) if miss_latencies_ms else 0.0,
            "p95": _p95_ms(miss_latencies_ms),
        },
        "speedup_factor_mean": speedup,
    }

    out_path = ROOT / "artifacts" / "cache_performance.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nCache Performance Measurement\n")
    print(f"Hit latency (mean):  {payload['hit_latency_ms']['mean']:.2f} ms")
    print(f"Miss latency (mean): {payload['miss_latency_ms']['mean']:.2f} ms")
    print(f"Speedup (mean):      {payload['speedup_factor_mean']:.2f}x")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()


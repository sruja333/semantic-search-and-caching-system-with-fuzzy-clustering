# AI/ML Engineer Task Solution

This repository implements a semantic search and semantic caching system on the 20 Newsgroups dataset, with fuzzy clustering and a FastAPI service.

## Key Features
- semantic search over ~20k Usenet documents
- fuzzy clustering with automatic cluster-count selection
- cluster-aware semantic cache for scalable lookup
- cache threshold tradeoff analysis
- boundary document detection via entropy
- FastAPI service with interactive demo UI
- Docker deployment

## 1. Problem Overview
The assignment asks for an end-to-end AI/ML system that does three things well:
- retrieves semantically relevant documents for natural-language queries
- groups documents into soft clusters (fuzzy membership, not only hard labels)
- reuses previous responses via a custom semantic cache (without external cache services)

This implementation delivers all required components and adds analysis artifacts to explain design tradeoffs (cluster quality, boundary ambiguity, threshold behavior, and latency impact).

## 2. System Architecture
```text
User Query
   -> Text Cleaning
   -> Embedding Model (TF-IDF + SVD / LSA)
   -> Cluster Detection (fuzzy membership)
   -> Cluster-Aware Cache Lookup (restricted to top-k query clusters)
      -> Hit  -> Return cached response
      -> Miss -> Vector Search
              -> Store in cache
              -> Return response
```

This cache partitioning keeps lookup efficient while preserving recall for ambiguous, multi-topic queries.

## Design Decisions
Several design choices were made to balance accuracy, interpretability, and reproducibility:
- Latent Semantic Analysis (TF-IDF + SVD) was chosen instead of transformer embeddings to keep the system lightweight and fully local.
- Fuzzy clustering is implemented by applying a softmax over centroid distances, producing a probability distribution over clusters.
- Cache lookup is restricted to the top-k query clusters, improving scalability while preserving recall for ambiguous queries.

Primary implementation files:
- `app/api.py`: FastAPI app and endpoints (`/query`, `/cache/stats`, `/cache`, `/demo`)
- `app/engine.py`: orchestration (artifact loading, query flow, cache integration)
- `app/vector_store.py`: embedding pipeline + nearest-neighbor retrieval
- `app/clustering.py`: cluster-count selection + fuzzy memberships + report generation
- `app/semantic_cache.py`: custom in-memory cluster-partitioned cache with eviction
- `app/cache_threshold.py`: threshold recommendation logic from scenario-based simulations
- `app/dataset.py`: dataset ingestion into typed `Document` objects
- `app/text_cleaning.py`: Usenet-specific cleaning and noise suppression
- `app/config.py`: global settings and artifact/data paths
- `app/static/demo.html`: lightweight web UI for live querying

Supporting files:
- `requirements.txt`: pinned dependency versions
- `scripts/build_index.py`: builds all model artifacts
- `scripts/run_cache_threshold_experiment.py`: threshold-vs-hit-rate experiment + plot
- `scripts/measure_cache_performance.py`: hit/miss latency benchmark
- `scripts/show_boundary_documents.py`: high-entropy boundary-document export
- `Dockerfile`: container image build recipe
- `docker-compose.yml`: local container orchestration
- `models/embeddings.npy`: saved embeddings artifact
- `notebooks/cluster_visualization.ipynb`: UMAP visual analysis notebook

Note: `app/__init__.py` is package wiring; `app/cache.py`, `app/main.py`, and `app/search.py` are currently placeholders.

## 3. Dataset Processing
Dataset used: **20 Newsgroups (Usenet posts)**.

Why this data is useful for this task:
- It is realistic and noisy (headers, quote chains, routing metadata), so preprocessing quality is genuinely tested.
- Topics overlap (for example religion/politics/firearms), which is ideal for demonstrating fuzzy clustering.
- It is large enough to show cache behavior at non-trivial scale.

What is done in preprocessing:
- read documents from category folders (`app/dataset.py`)
- preserve only intent-bearing headers (`Subject`, `Keywords`) and drop high-noise transport headers
- remove quote chains/signatures
- normalize contractions, remove emails/URLs/non-alphabetic noise
- prune very short tokens and known Usenet filler tokens

Outcome from `artifacts/build_report.json`:
- indexed documents: **19,829**

How this helps understanding:
- cleaned text preserves topical language while reducing routing and conversational noise
- fuzzy cluster boundaries become more meaningful because noise is reduced but ambiguity is preserved

## 4. Embedding Model Choice
Embedding strategy (`app/vector_store.py`):
- `TfidfVectorizer` with unigram + bigram features
- `TruncatedSVD` (Latent Semantic Analysis, LSA) to project sparse TF-IDF vectors into a dense semantic space (256 dims)
- L2 normalization for cosine-based similarity retrieval

Why this model choice is appropriate:
- local and reproducible (no external API dependency)
- strong baseline for long-form topic retrieval on news/forum text
- efficient on CPU and easy to persist/deploy

Stored artifact:
- `artifacts/vector_store.joblib`
- metadata persisted with each vector (`doc_id`, `label`, `preview`) to support efficient downstream filtering and analysis.

## 5. Clustering Analysis
Clustering approach (`app/clustering.py`):
- hard centers via `MiniBatchKMeans`
- soft/fuzzy memberships via softmax over centroid distances

Cluster-count selection:
- candidates tested: `{12, 16, 20, 24, 28, 32}`
- metrics used: silhouette, Calinski-Harabasz, Davies-Bouldin
- combined normalized score chooses final `k`

Selected result:
- selected clusters: **28** (`artifacts/cluster_report.json`)

Analysis artifacts:
- per-cluster dominant labels and representative documents
- boundary documents and global uncertain documents (entropy-based)
- notebook visualizations with UMAP projections and cluster centers

Example cluster topics from c-TF-IDF keyword extraction (`notebooks/cluster_visualization.ipynb`):
- Cluster 24 -> orbit, mars, planetary, hst, planets, mercury, funding, temperature, craft, viking
- Cluster 01 -> catholic, biblical, worship, doctrine, orthodox, passage, marriage, divine, arrogance, luke
- Cluster 04 -> isa, vga, vesa, slots, slot, stealth, controllers, macs, nec, parity
- Cluster 20 -> anonymous ftp, gopher, available anonymous, msdos, edu pub, tcp, download, readme, directories, password

Files:
- `artifacts/cluster_report.json`
- `artifacts/boundary_documents.json`
- `notebooks/cluster_visualization.ipynb`

## 6. Semantic Cache Design
Cache design (`app/semantic_cache.py`):
- custom cache entries store query text, embedding, dominant cluster, top clusters, and response payload
- cache index is partitioned by cluster ID (`cluster_id -> cache keys`)
- lookup probes only likely cluster partitions (top-3 query clusters), not the full cache
- this maintains recall for ambiguous queries while still reducing search space
- LRU-like eviction via ordered dictionary when max size is exceeded
- runtime stats exposed (`hit_count`, `miss_count`, `hit_rate`)

Why this design matters:
- scales better than global linear scan as cache grows
- practical complexity improves from roughly `O(n)` to approximately `O(n/k)` under balanced clusters
- integrates naturally with fuzzy clustering output

## 7. Threshold Experiments
Two complementary threshold analyses are included.

### 7.1 Operational Hit-Rate Tradeoff (`scripts/run_cache_threshold_experiment.py`)
Generated files:
- `artifacts/cache_threshold_experiment.json`
- `artifacts/cache_threshold_tradeoff.png`

Measured results:

| Threshold | Hit Rate | Avg Similarity |
|---|---:|---:|
| 0.80 | 27.8% | 0.869 |
| 0.85 | 19.8% | 0.898 |
| 0.90 | 10.4% | 0.932 |
| 0.95 | 3.6% | 0.966 |

Interpretation:
- lower threshold -> higher reuse, lower strictness
- higher threshold -> lower reuse, higher semantic precision

### 7.2 Scenario-Based Recommendation (`app/cache_threshold.py`)
This study simulates:
- exact duplicate pairs
- same-topic rephrases
- different-topic negatives

Output:
- `artifacts/cache_threshold_report.json`

Recommended threshold from build report:
- **0.80**

## 8. API Endpoints
Base service: `uvicorn app.api:app`

State management:
- the service uses FastAPI lifespan initialization and `app.state.engine` so the loaded vector DB, cluster model, and cache are reused safely across requests.

- `GET /`
  - health/status message
- `GET /demo`
  - simple HTML demo UI
- `GET /docs`
  - interactive Swagger UI
- `POST /query`
  - input: `{ "query": "space shuttle launch and nasa mission" }`
  - output fields: `query`, `cache_hit`, `matched_query`, `similarity_score`, `result`, `dominant_cluster`
- `GET /cache/stats`
  - output: total entries, hit/miss count, hit rate
- `DELETE /cache`
  - clears cache and resets counters

Example `POST /query` response (truncated):
```json
{
  "query": "space shuttle launch and nasa mission",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "dominant_cluster": 24,
  "result": {
    "top_matches": [
      {
        "doc_id": "sci.space/61430",
        "label": "sci.space",
        "score": 0.8309
      }
    ]
  }
}
```

## 9. Running the System
### 9.1 Local setup (PowerShell)
```powershell
cd c:\Users\sruja\OneDrive\Desktop\trademarkia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 9.2 Build artifacts (first time or after model/cleaning changes)
```powershell
python scripts/build_index.py
```

### 9.3 Start API
```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000/demo`
- `http://127.0.0.1:8000/docs`

### 9.4 Quick terminal test of output
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/query" -ContentType "application/json" -Body '{"query":"space shuttle launch and nasa mission"}'
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/cache/stats"
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8000/cache"
```

### 9.5 Run analysis scripts
```powershell
python scripts/run_cache_threshold_experiment.py
python scripts/measure_cache_performance.py
python scripts/show_boundary_documents.py
```

Generated outputs:
- `artifacts/cache_threshold_experiment.json`
- `artifacts/cache_threshold_tradeoff.png`
- `artifacts/cache_performance.json`
- `artifacts/boundary_documents.json`

### 9.6 Notebook visualization
Open and run:
- `notebooks/cluster_visualization.ipynb`

### 9.7 Docker
Build and run:
```powershell
docker build -t trademarkia-semantic-cache .
docker run --rm -p 8000:8000 trademarkia-semantic-cache
```

Or compose:
```powershell
docker compose up --build
```

Then open:
- `http://localhost:8000/demo`
- `http://localhost:8000/docs`

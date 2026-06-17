# Fairness-Aware Conversational Recommender System (FA-CRS)

Group project for the course **Human-Centered Artificial Intelligence**.

This system builds a fairness-aware movie recommender on MovieLens-1M by combining a heterogeneous knowledge graph with LightGCN and FA\*IR post-hoc reranking. It targets two forms of bias: **director gender bias** (under-representation of female-directed films) and **production country bias** (under-representation of non-western cinema). Each recommendation is accompanied by a natural language explanation indicating whether it was selected for relevance, gender diversity, or geographic diversity.

---

## Repository Structure

```
├── data_prep.py            # Step 1: enrich MovieLens with TMDb metadata + fairness attributes
├── lightgcn_baseline.py    # Step 2: train vanilla LightGCN baseline
├── heterogeneous_kg.py     # Step 3: build KG + retrain LightGCN on enriched graph
├── fair_rerank.py          # Step 4: FA*IR reranking + explanation generation
├── debug_fair.py           # Diagnostic: verify candidate pool and protected-group injection
├── results.py              # Step 5: generate paper figures and comparison table
├── ml-1m.zip               # MovieLens-1M dataset (raw)
├── outputs/                # All generated results (created at runtime)
│   ├── baseline/
│   ├── kg/
│   ├── fair/
│   └── figures/
└── README.md
```

---

## How It Works

The system runs as a five-stage pipeline on MovieLens-1M.

**1. Data preparation (`data_prep.py`)** — Loads the MovieLens-1M movie catalogue and enriches each title via the TMDb API, fetching the director name and production country codes. Director gender is inferred from the first name using `gender-guesser`, and production country is classified as western or non-western using an ISO 3166-1 country list. The enriched dataset is saved to `data/movies_enriched.csv` with an intermediate cache for resumable API calls.

**2. Baseline (`lightgcn_baseline.py`)** — Trains a standard LightGCN model on the bipartite user–movie interaction graph (implicit feedback from ratings ≥ 4). Evaluates on NDCG@10, Precision@10, Recall@10, and fairness metrics (SPD, EOD) for both director gender and production region. Results saved to `outputs/baseline/baseline_results.json`.

**3. Heterogeneous knowledge graph (`heterogeneous_kg.py`)** — Extends the interaction graph into a full heterogeneous KG by adding director, gender, region, and genre nodes with typed edges. Retrains LightGCN on this enriched graph, allowing the model to propagate fairness-relevant signals during message passing. Results saved to `outputs/kg/`.

**4. FA\*IR reranking + explanations (`fair_rerank.py`)** — Loads the KG model weights and applies FA\*IR reranking to enforce minimum protected-group proportions in the top-10 recommendations. Reranking is applied sequentially for director gender and then production region. A fairness strength parameter `p` is swept across five values (0.1–0.5) to produce the Fairness-Utility Tradeoff (FUT) curve. Each recommendation is tagged with a template-based explanation indicating whether it was selected for relevance, gender diversity, or geographic diversity. Results saved to `outputs/fair/`.

**5. Figures and paper table (`results.py`)** — Reads the three JSON result files and generates all paper-ready outputs: a formatted comparison table across the three systems, the combined FUT curve figure (NDCG vs SPD across p values), a bias comparison bar chart, and an accuracy comparison bar chart. All figures saved to `outputs/figures/`.

---

## Installation

Python 3.9+ is recommended.

```bash
pip install torch torch-geometric pandas numpy scikit-learn tqdm requests gender-guesser matplotlib
```

For GPU training, install PyTorch with the appropriate CUDA version from [pytorch.org](https://pytorch.org) before running the above.

---

## Setup

**1. Unzip MovieLens-1M**

```bash
unzip ml-1m.zip -d data/ml-1m
```

Expected files: `data/ml-1m/ratings.dat`, `data/ml-1m/movies.dat`, `data/ml-1m/users.dat`

**2. Add your TMDb API key**

Open `data_prep.py` and paste your key on this line:

```python
TMDB_API_KEY = "your_key_here"
```

Get a free key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

---

## Running the Pipeline

Run each script in order. Each step depends on the outputs of the previous one.

```bash
# Step 1: enrich movie metadata (~30–60 min due to TMDb rate limits; resumable)
python data_prep.py

# Step 2: train baseline LightGCN
python lightgcn_baseline.py

# Step 3: build KG and retrain
python heterogeneous_kg.py

# Step 4: FA*IR reranking + explanations
python fair_rerank.py

# Step 5: generate figures and paper table
python results.py
```

---

## Outputs

| File | Description |
|------|-------------|
| `data/movies_enriched.csv` | MovieLens-1M enriched with director, gender, and region |
| `outputs/baseline/baseline_results.json` | Baseline accuracy + fairness metrics |
| `outputs/kg/kg_results.json` | KG model accuracy + fairness metrics |
| `outputs/kg/best_model_kg.pt` | Saved KG model weights |
| `outputs/fair/fut_curve.json` | FUT curve data across p values |
| `outputs/fair/fair_results.json` | Full comparison table data |
| `outputs/fair/example_recommendations.txt` | Sample recommendations with explanations |
| `outputs/figures/results_table.txt` | Formatted Table 1 for paper |
| `outputs/figures/fut_curve_combined.png` | Main FUT curve figure |
| `outputs/figures/bias_comparison.png` | SPD comparison across systems |
| `outputs/figures/accuracy_comparison.png` | NDCG/Precision/Recall across systems |

---

## Evaluation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| NDCG@10 | Accuracy | Normalized discounted cumulative gain at rank 10 |
| Precision@10 | Accuracy | Fraction of top-10 recommendations that are relevant |
| Recall@10 | Accuracy | Fraction of relevant items appearing in top-10 |
| SPD | Fairness | Statistical Parity Difference — exposure gap between groups |
| EOD | Fairness | Equal Opportunity Difference — true positive rate gap between groups |

Fairness is measured separately for director gender (female vs. male) and production region (non-western vs. western). Lower absolute SPD and EOD indicate fairer recommendations.

---

## Key Design Decisions

- **Sensitive attributes**: director gender (inferred from first name) and production country (western / non-western based on ISO 3166-1 codes).
- **Graph enrichment**: the KG adds director → gender and movie → region edges so that LightGCN can propagate group membership during neighbourhood aggregation.
- **FA\*IR**: a position-aware reranking algorithm that enforces a minimum protected-group proportion `p` at each rank position. Applied twice in sequence — first for gender, then for region.
- **Protected pool injection**: the candidate pool for each user is explicitly seeded with the top-scoring female-directed and non-western films they haven't seen, ensuring FA\*IR always has protected items to promote.
- **FUT curve**: `p` is varied from 0.1 to 0.5 to trace the tradeoff between accuracy (NDCG) and fairness (SPD). The primary reported result uses `p = 0.3`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torch-geometric` | LightGCN model and graph utilities |
| `pandas` / `numpy` | Data handling |
| `requests` | TMDb API calls |
| `gender-guesser` | Director gender inference |
| `tqdm` | Progress bars |
| `matplotlib` | Figure generation |
| `scikit-learn` | Train/val/test split utilities |

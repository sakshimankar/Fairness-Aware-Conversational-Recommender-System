"""
FA-CRS Day 10-12: FA*IR Reranking + Explanation Module
-------------------------------------------------------
Loads the trained KG model from Day 7-9 and applies FA*IR reranking
to enforce fairness constraints on the top-K recommendations.

FA*IR algorithm:
  - Works through recommendation positions one by one
  - At each position checks if the proportion of the protected group
    (female-directed, non-western) meets a minimum threshold p
  - If not, pulls in the highest-scoring movie from that group
  - Controlled by two parameters:
      p     : minimum proportion for protected group (default 0.3)
      alpha : significance level for the statistical test (default 0.1)

Explanation module:
  - Template-based strings explaining each recommendation
  - Tags each movie as relevance-driven or fairness-driven

Output:
  - outputs/fair/fair_results.json
  - outputs/fair/full_comparison_table.json
  - outputs/fair/example_recommendations.txt  (sample explanations)

Requirements: same as Day 7-9
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR      = "data"
KG_OUTPUT_DIR = "outputs/kg"
OUTPUT_DIR    = "outputs/fair"
EMBEDDING_DIM = 32
NUM_LAYERS    = 2
MIN_RATING    = 4
TOP_K         = 10
RANDOM_SEED   = 42

# FA*IR parameters
# p: minimum proportion of protected group in top-K
# Try multiple values to generate the FUT curve for your paper
FAIR_P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]
FAIR_ALPHA    = 0.1   # significance level

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies_enriched.csv"))

    pos = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()

    # ── ADD THIS: Keep only active users (>=20 interactions) ──
    user_counts  = pos["user_id"].value_counts()
    active_users = user_counts[user_counts >= 20].index
    pos = pos[pos["user_id"].isin(active_users)].copy()

    user_ids  = sorted(pos["user_id"].unique())
    movie_ids = sorted(pos["movie_id"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}

    pos["user_idx"]  = pos["user_id"].map(user2idx)
    pos["movie_idx"] = pos["movie_id"].map(movie2idx)

    n_users  = len(user_ids)
    n_movies = len(movie_ids)

    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    movies["director"]        = movies["director"].fillna("Unknown Director")
    movies["director_gender"] = movies["director_gender"].fillna("unknown")
    movies["region"]          = movies["region"].fillna("unknown")
    movies["genres"]          = movies["genres"].fillna("Unknown")

    # ── ADD THIS: Keep only directors with 2+ movies ──
    director_counts    = movies["director"].value_counts()
    frequent_directors = set(director_counts[director_counts >= 2].index)
    movies["director"] = movies["director"].apply(
        lambda d: d if d in frequent_directors else "Unknown Director")

    return pos, movies, user2idx, movie2idx, n_users, n_movies

def split_data(pos):
    train_rows, val_rows, test_rows = [], [], []
    for _, group in pos.groupby("user_idx"):
        items = group["movie_idx"].tolist()
        if len(items) < 3:
            train_rows.extend([(group["user_idx"].iloc[0], m) for m in items])
            continue
        n_val  = max(1, int(0.1 * len(items)))
        n_test = max(1, int(0.1 * len(items)))
        train  = items[:-(n_val + n_test)]
        val    = items[-(n_val + n_test):-n_test]
        test   = items[-n_test:]
        uid    = group["user_idx"].iloc[0]
        train_rows.extend([(uid, m) for m in train])
        val_rows.extend([(uid, m) for m in val])
        test_rows.extend([(uid, m) for m in test])

    train_df = pd.DataFrame(train_rows, columns=["user_idx", "movie_idx"])
    val_df   = pd.DataFrame(val_rows,   columns=["user_idx", "movie_idx"])
    test_df  = pd.DataFrame(test_rows,  columns=["user_idx", "movie_idx"])
    return train_df, val_df, test_df


# ─── REBUILD KG (same as Day 7-9) ─────────────────────────────────────────────

def build_kg(train_df, movies, n_users, n_movies):
    directors    = sorted(movies["director"].unique())
    dir2idx      = {d: i for i, d in enumerate(directors)}
    n_directors  = len(directors)
    dir_offset   = n_users + n_movies

    genders       = ["female", "male", "unknown"]
    gender2idx    = {g: i for i, g in enumerate(genders)}
    n_genders     = len(genders)
    gender_offset = dir_offset + n_directors

    regions       = ["western", "non-western", "unknown"]
    region2idx    = {r: i for i, r in enumerate(regions)}
    n_regions     = len(regions)
    region_offset = gender_offset + n_genders

    all_genres = set()
    for g in movies["genres"]:
        for genre in g.split("|"):
            all_genres.add(genre.strip())
    genres       = sorted(all_genres)
    genre2idx    = {g: i for i, g in enumerate(genres)}
    n_genres     = len(genres)
    genre_offset = region_offset + n_regions

    u_idx    = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m_idx    = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)
    um_src   = torch.cat([u_idx, m_idx])
    um_dst   = torch.cat([m_idx, u_idx])

    movie_nodes = torch.tensor(movies["movie_idx"].values + n_users, dtype=torch.long)
    dir_nodes   = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    md_src = torch.cat([movie_nodes, dir_nodes])
    md_dst = torch.cat([dir_nodes, movie_nodes])

    dir_nodes_g = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    gen_nodes   = torch.tensor([gender2idx[g] + gender_offset for g in movies["director_gender"]], dtype=torch.long)
    dg_src = torch.cat([dir_nodes_g, gen_nodes])
    dg_dst = torch.cat([gen_nodes, dir_nodes_g])

    reg_nodes = torch.tensor([region2idx[r] + region_offset for r in movies["region"]], dtype=torch.long)
    mr_src = torch.cat([movie_nodes, reg_nodes])
    mr_dst = torch.cat([reg_nodes, movie_nodes])

    mg_srcs, mg_dsts = [], []
    for _, row in movies.iterrows():
        m_node = int(row["movie_idx"]) + n_users
        for genre in row["genres"].split("|"):
            genre = genre.strip()
            if genre in genre2idx:
                g_node = genre2idx[genre] + genre_offset
                mg_srcs.extend([m_node, g_node])
                mg_dsts.extend([g_node, m_node])

    mg_src = torch.tensor(mg_srcs, dtype=torch.long)
    mg_dst = torch.tensor(mg_dsts, dtype=torch.long)

    all_src    = torch.cat([um_src, md_src, dg_src, mr_src, mg_src])
    all_dst    = torch.cat([um_dst, md_dst, dg_dst, mr_dst, mg_dst])
    edge_index = torch.stack([all_src, all_dst], dim=0).to(device)

    n_total = n_users + n_movies + n_directors + n_genders + n_regions + n_genres
    return edge_index, n_total


# ─── LIGHTGCN MODEL (same as Day 7-9) ────────────────────────────────────────

class LightGCNModel(nn.Module):
    def __init__(self, n_total, n_users, n_movies, embedding_dim, num_layers):
        super().__init__()
        self.n_users  = n_users
        self.n_movies = n_movies
        self.embedding = nn.Embedding(n_total, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.lgcn = LightGCN(n_total, embedding_dim, num_layers)
        self.lgcn.embedding = self.embedding

    def forward(self, edge_index):
        x = self.lgcn.get_embedding(edge_index)
        return x[:self.n_users], x[self.n_users: self.n_users + self.n_movies]


# ─── GET RAW SCORES ───────────────────────────────────────────────────────────

def get_scores(model, edge_index, train_df, n_users, n_movies,
               candidate_k=50, movie_gender=None, movie_region=None):
    """
    Returns candidate (movie_idx, score) pairs per user BEFORE reranking.

    Injects protected group movies into every user's candidate pool so
    FA*IR always has something to promote. Without this, the pool contains
    almost no female-directed or non-western movies and reranking has no effect.
    """
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)

    seen          = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    female_movies = set(m for m, g in movie_gender.items() if g == "female")     if movie_gender else set()
    nonwestern_movies = set(m for m, r in movie_region.items() if r == "non-western") if movie_region else set()

    candidates = {}
    SCORE_BATCH = 1000
    for start in range(0, n_users, SCORE_BATCH):
        end = min(start + SCORE_BATCH, n_users)
        batch_scores = torch.matmul(user_emb[start:end], movie_emb.T).cpu().numpy()

        for i, u in enumerate(range(start, end)):
            seen_u = seen.get(u, set())
            s = batch_scores[i].copy()
            for m in seen_u:
                if m < len(s):
                    s[m] = -np.inf

            top  = np.argpartition(s, -candidate_k)[-candidate_k:]
            top  = top[np.argsort(s[top])[::-1]]
            pool = set(top.tolist())

            # Inject protected movies so FA*IR has something to promote
            female_unseen = sorted(
                [m for m in female_movies if m not in seen_u and m < len(s)],
                key=lambda m: s[m], reverse=True)
            for m in female_unseen[:10]:
                pool.add(m)

            nw_unseen = sorted(
                [m for m in nonwestern_movies if m not in seen_u and m < len(s)],
                key=lambda m: s[m], reverse=True)
            for m in nw_unseen[:10]:
                pool.add(m)

            pool_list = sorted(pool, key=lambda m: s[m] if s[m] > -1e8 else -1e9, reverse=True)
            candidates[u] = [(int(m), float(s[m])) for m in pool_list]

    return candidates   # ← single correct return


# ─── FA*IR RERANKING ──────────────────────────────────────────────────────────

def fair_rerank(candidates, movie_attr, protected_val, p, k=TOP_K):
    """
    Simplified FA*IR-style reranking.

    At each position, checks if the proportion of protected items so far
    is below p. If yes, forces the next best protected item into that slot.
    Otherwise places the highest-scoring item regardless of group.

    candidates    : list of (movie_idx, score) sorted by score descending
    movie_attr    : dict {movie_idx: attribute_value}
    protected_val : e.g. "female" or "non-western"
    p             : minimum target proportion for protected group
    k             : final list size
    """
    protected   = [(m, s) for m, s in candidates if movie_attr.get(m) == protected_val]
    unprotected = [(m, s) for m, s in candidates if movie_attr.get(m) != protected_val]

    result       = []
    result_flags = []
    p_ptr        = 0
    u_ptr        = 0

    for pos in range(k):
        n_placed    = pos  # items placed so far
        n_protected = sum(1 for f in result_flags if f == True)

        # How many protected do we need by this position to meet proportion p?
        needed = math.ceil(p * (pos + 1))

        if n_protected < needed and p_ptr < len(protected):
            # Force a protected item
            result.append(protected[p_ptr][0])
            result_flags.append(True)
            p_ptr += 1
        else:
            # Take whichever is higher scoring
            take_protected = (
                p_ptr < len(protected) and
                (u_ptr >= len(unprotected) or
                 protected[p_ptr][1] >= unprotected[u_ptr][1])
            )
            if take_protected:
                result.append(protected[p_ptr][0])
                result_flags.append(False)
                p_ptr += 1
            elif u_ptr < len(unprotected):
                result.append(unprotected[u_ptr][0])
                result_flags.append(False)
                u_ptr += 1

        if len(result) == k:
            break

    return result, result_flags


def rerank_all_users(candidates, movie_gender, movie_region, p_gender, p_region):
    """
    Apply FA*IR twice: once for gender, once for region.
    Gender reranking is applied first, then region reranking on top.
    Returns recs dict and flags dict.
    """
    recs  = {}
    flags = {}  # (user_idx) -> list of ('gender'|'region'|'relevance') per position

    for u, cands in candidates.items():
        # Step 1: gender fairness
        reranked_gender, gender_flags = fair_rerank(
            cands, movie_gender, "female", p_gender)

        # Rebuild candidate list in new order for region pass
        reranked_scores = {m: s for m, s in cands}
        reranked_cands  = [(m, reranked_scores.get(m, -np.inf)) for m in reranked_gender]
        # Add any remaining candidates not yet in list
        seen_set = set(reranked_gender)
        for m, s in cands:
            if m not in seen_set:
                reranked_cands.append((m, s))

        # Step 2: region fairness on top of gender-reranked list
        reranked_region, region_flags = fair_rerank(
            reranked_cands, movie_region, "non-western", p_region)

        # Combine flags: label each position
        combined_flags = []
        for i, m in enumerate(reranked_region):
            if i < len(region_flags) and region_flags[i]:
                combined_flags.append("region")
            elif i < len(gender_flags) and gender_flags[i]:
                combined_flags.append("gender")
            else:
                combined_flags.append("relevance")

        recs[u]  = reranked_region
        flags[u] = combined_flags

    return recs, flags


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def precision_recall_ndcg(recs, ground_truth_df):
    gt = ground_truth_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    precisions, recalls, ndcgs = [], [], []
    for u, rec_list in recs.items():
        if u not in gt:
            continue
        actual = gt[u]
        hits   = [1 if m in actual else 0 for m in rec_list[:TOP_K]]
        precision = sum(hits) / TOP_K
        recall    = sum(hits) / len(actual) if actual else 0
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), TOP_K)))
        ndcg = dcg / idcg if idcg > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


def compute_fairness_metrics(recs, test_df, movies, attribute_col, group_a, group_b):
    movie_group = movies.set_index("movie_idx")[attribute_col].to_dict()
    gt          = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    spd_list, eod_list = [], []
    for u, rec_list in recs.items():
        rec_set = set(rec_list)
        rec_a   = sum(1 for m in rec_set if movie_group.get(m) == group_a)
        rec_b   = sum(1 for m in rec_set if movie_group.get(m) == group_b)
        total   = rec_a + rec_b
        if total == 0:
            continue
        spd_list.append(rec_a / total - rec_b / total)
        relevant = gt.get(u, set())
        rel_a = sum(1 for m in relevant if movie_group.get(m) == group_a)
        rel_b = sum(1 for m in relevant if movie_group.get(m) == group_b)
        hit_a = sum(1 for m in relevant if m in rec_set and movie_group.get(m) == group_a)
        hit_b = sum(1 for m in relevant if m in rec_set and movie_group.get(m) == group_b)
        tpr_a = hit_a / rel_a if rel_a > 0 else None
        tpr_b = hit_b / rel_b if rel_b > 0 else None
        if tpr_a is not None and tpr_b is not None:
            eod_list.append(tpr_a - tpr_b)
    return (np.mean(spd_list) if spd_list else 0.0,
            np.mean(eod_list) if eod_list else 0.0)


# ─── EXPLANATION MODULE ───────────────────────────────────────────────────────

def generate_explanation(movie_row, flag, rank):
    """
    Template-based explanation for a single recommendation.
    flag: 'relevance', 'gender', or 'region'
    """
    title  = movie_row.get("title", "This movie")
    director = movie_row.get("director", "the director")
    genres = movie_row.get("genres", "").replace("|", ", ")
    gender = movie_row.get("director_gender", "unknown")
    region = movie_row.get("region", "unknown")

    if flag == "relevance":
        return (f"#{rank} {title} — Recommended based on your viewing history. "
                f"Genre: {genres}. Directed by {director}.")

    elif flag == "gender":
        return (f"#{rank} {title} — Highlighted to support gender diversity in recommendations. "
                f"Directed by {director} ({gender}-directed). Genre: {genres}.")

    elif flag == "region":
        return (f"#{rank} {title} — Highlighted to support geographic diversity. "
                f"This is a {region} production directed by {director}. Genre: {genres}.")

    return f"#{rank} {title}"


def generate_user_explanations(user_idx, rec_list, flags, movies_indexed):
    lines = [f"Recommendations for User {user_idx}:", "-" * 50]
    for rank, (movie_idx, flag) in enumerate(zip(rec_list, flags), 1):
        if movie_idx in movies_indexed.index:
            row = movies_indexed.loc[movie_idx]
            row_dict = row.to_dict() if hasattr(row, "to_dict") else {}
        else:
            row_dict = {}
        lines.append(generate_explanation(row_dict, flag, rank))
    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Load data
    pos, movies, user2idx, movie2idx, n_users, n_movies = load_data()
    train_df, val_df, test_df = split_data(pos)

    # Rebuild KG and model
    print("Rebuilding KG...")
    edge_index, n_total = build_kg(train_df, movies, n_users, n_movies)

    model = LightGCNModel(n_total, n_users, n_movies, EMBEDDING_DIM, NUM_LAYERS).to(device)
    model_path = os.path.join(KG_OUTPUT_DIR, "best_model_kg.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded KG model weights.")

    # Movie attribute lookups (must be defined before get_scores)
    movie_gender   = movies.set_index("movie_idx")["director_gender"].to_dict()
    movie_region   = movies.set_index("movie_idx")["region"].to_dict()
    movies_indexed = movies.set_index("movie_idx")

    # Get candidate scores — injects protected movies into pool
    print("Scoring candidates (expanding pool with protected group movies)...")
    candidates = get_scores(model, edge_index, train_df, n_users, n_movies,
                            candidate_k=50,
                            movie_gender=movie_gender,
                            movie_region=movie_region)

    # ── Run FA*IR across multiple p values to generate FUT curve ──
    print("\nRunning FA*IR reranking across p values...")
    fut_curve = []

    for p in FAIR_P_VALUES:
        recs, flags = rerank_all_users(
            candidates, movie_gender, movie_region,
            p_gender=p, p_region=p
        )
        prec, rec, ndcg = precision_recall_ndcg(recs, test_df)
        spd_g, eod_g    = compute_fairness_metrics(recs, test_df, movies, "director_gender", "female", "male")
        spd_r, eod_r    = compute_fairness_metrics(recs, test_df, movies, "region", "non-western", "western")

        fut_curve.append({
            "p":             p,
            "ndcg_at_10":    round(ndcg, 4),
            "precision_at_10": round(prec, 4),
            "recall_at_10":  round(rec, 4),
            "gender_spd":    round(spd_g, 4),
            "gender_eod":    round(eod_g, 4),
            "region_spd":    round(spd_r, 4),
            "region_eod":    round(eod_r, 4),
        })

        print(f"p={p:.1f} | NDCG: {ndcg:.4f} | G-SPD: {spd_g:.4f} | R-SPD: {spd_r:.4f}")

    # Save FUT curve
    with open(os.path.join(OUTPUT_DIR, "fut_curve.json"), "w") as f:
        json.dump(fut_curve, f, indent=2)
    print(f"\nFUT curve saved.")

    # ── Use p=0.3 as the primary result ──
    primary_p = 0.3
    primary   = next(r for r in fut_curve if r["p"] == primary_p)

    # ── Full comparison table ──
    baseline_path = "outputs/baseline/baseline_results.json"
    kg_path       = "outputs/kg/kg_results.json"

    baseline = json.load(open(baseline_path)) if os.path.exists(baseline_path) else {}
    kg       = json.load(open(kg_path))       if os.path.exists(kg_path)       else {}

    print(f"\n{'='*70}")
    print(f"FULL COMPARISON TABLE (p={primary_p} for FA*IR)")
    print(f"{'='*70}")
    print(f"{'Metric':<20} {'Baseline':>10} {'KG':>10} {'KG+FA*IR':>10} {'Δ Fair':>10}")
    print("-" * 64)

    metric_keys = [
        ("NDCG@10",      "ndcg_at_10"),
        ("Precision@10", "precision_at_10"),
        ("Recall@10",    "recall_at_10"),
        ("Gender SPD",   "gender_spd"),
        ("Gender EOD",   "gender_eod"),
        ("Region SPD",   "region_spd"),
        ("Region EOD",   "region_eod"),
    ]

    comparison = {}
    for label, key in metric_keys:
        b  = baseline.get(key, 0)
        k  = kg.get(key, 0)
        f  = primary.get(key, 0)
        delta = f - k
        print(f"{label:<20} {b:>10.4f} {k:>10.4f} {f:>10.4f} {delta:>+10.4f}")
        comparison[label] = {"baseline": b, "kg": k, "kg_fair": f, "delta_fair": delta}

    # Save full comparison
    full_results = {
        "primary_p":       primary_p,
        "fair_alpha":      FAIR_ALPHA,
        "comparison_table": comparison,
        "fut_curve":       fut_curve,
    }
    with open(os.path.join(OUTPUT_DIR, "fair_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    # ── Generate example explanations for 5 users ──
    print(f"\n{'='*70}")
    print("EXAMPLE RECOMMENDATIONS WITH EXPLANATIONS")
    print(f"{'='*70}")

    # Re-run FA*IR at p=0.3 to get flags for explanations
    recs, flags = rerank_all_users(
        candidates, movie_gender, movie_region,
        p_gender=primary_p, p_region=primary_p
    )

    explanation_lines = []
    for u in list(recs.keys())[:5]:
        block = generate_user_explanations(u, recs[u], flags[u], movies_indexed)
        print(block)
        print()
        explanation_lines.append(block)
        explanation_lines.append("")

    with open(os.path.join(OUTPUT_DIR, "example_recommendations.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(explanation_lines))

    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("\nNext step: Day 13-15 write-up. Use fair_results.json for your results table.")


if __name__ == "__main__":
    main()

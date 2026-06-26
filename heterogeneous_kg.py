"""
FA-CRS Day 7-9: Heterogeneous Knowledge Graph + LightGCN
---------------------------------------------------------
Builds on Day 4-6 baseline by enriching the graph with:
  - Director nodes
  - Gender nodes  (female, male, unknown)
  - Region nodes  (western, non-western, unknown)
  - Genre nodes

New edges added:
  - movie  -> director
  - director -> gender
  - movie  -> region
  - movie  -> genre

Same LightGCN model trained on this richer graph.
Evaluation uses same metrics so results are directly comparable.

Output: outputs/kg/kg_results.json  (slot into comparison table alongside baseline)

Requirements: same as Day 4-6
    pip install torch torch-geometric pandas numpy scikit-learn tqdm
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
import gc

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR      = "data"
OUTPUT_DIR    = "outputs/kg"
EMBEDDING_DIM = 32     # reduced from 64 to save GPU memory
NUM_LAYERS    = 2      # reduced from 3 to save GPU memory
LEARNING_RATE = 1e-3
EPOCHS        = 30
MIN_RATING    = 4
TOP_K         = 10
RANDOM_SEED   = 42
SCORE_BATCH   = 512    # users per batch during evaluation to avoid OOM

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies_enriched.csv"))

    # Positive interactions only
    pos = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()

    # Keep only active users (>=20 interactions) to reduce graph size
    user_counts  = pos["user_id"].value_counts()
    active_users = user_counts[user_counts >= 20].index
    pos = pos[pos["user_id"].isin(active_users)].copy()
    print(f"After filtering inactive users: {pos['user_id'].nunique()} users")

    # Re-index users and movies
    user_ids  = sorted(pos["user_id"].unique())
    movie_ids = sorted(pos["movie_id"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}

    pos["user_idx"]  = pos["user_id"].map(user2idx)
    pos["movie_idx"] = pos["movie_id"].map(movie2idx)

    n_users  = len(user_ids)
    n_movies = len(movie_ids)

    # Attach movie_idx to enriched movies
    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)

    # Fill missing metadata
    movies["director"]        = movies["director"].fillna("Unknown Director")
    movies["director_gender"] = movies["director_gender"].fillna("unknown")
    movies["region"]          = movies["region"].fillna("unknown")
    movies["genres"]          = movies["genres"].fillna("Unknown")

    # Keep only directors with 2+ movies to reduce director node count
    director_counts    = movies["director"].value_counts()
    frequent_directors = set(director_counts[director_counts >= 2].index)
    movies["director"] = movies["director"].apply(
        lambda d: d if d in frequent_directors else "Unknown Director")
    print(f"Director nodes after filtering singletons: {movies['director'].nunique()}")

    print(f"Users: {n_users}, Movies: {n_movies}")
    print(f"Positive interactions: {len(pos)}")
    return pos, movies, user2idx, movie2idx, n_users, n_movies


# ─── TRAIN/VAL/TEST SPLIT ─────────────────────────────────────────────────────

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
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


# ─── BUILD HETEROGENEOUS KNOWLEDGE GRAPH ──────────────────────────────────────

def build_kg(train_df, movies, n_users, n_movies):
    """
    Node index layout (all offset so they don't overlap):
      [0,          n_users)         -> user nodes
      [n_users,    n_users+n_movies) -> movie nodes
      then appended in order:
        director nodes
        gender nodes   (female, male, unknown)
        region nodes   (western, non-western, unknown)
        genre nodes

    Returns:
      edge_index : [2, E] tensor with ALL edges (both directions)
      n_total    : total number of nodes
      node_map   : dict with offset info (for debugging)
    """

    # ── 1. User-Movie edges (same as bipartite baseline) ──
    u_idx = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m_idx = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)

    um_src = torch.cat([u_idx, m_idx])
    um_dst = torch.cat([m_idx, u_idx])

    # ── 2. Director nodes ──
    directors     = sorted(movies["director"].unique())
    dir2idx       = {d: i for i, d in enumerate(directors)}
    n_directors   = len(directors)
    dir_offset    = n_users + n_movies
    print(f"Director nodes: {n_directors}")

    # movie -> director edges
    movie_nodes = torch.tensor(
        movies["movie_idx"].values + n_users, dtype=torch.long)
    dir_nodes   = torch.tensor(
        [dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)

    md_src = torch.cat([movie_nodes, dir_nodes])
    md_dst = torch.cat([dir_nodes, movie_nodes])

    # ── 3. Gender nodes ──
    genders       = ["female", "male", "unknown"]
    gender2idx    = {g: i for i, g in enumerate(genders)}
    n_genders     = len(genders)
    gender_offset = dir_offset + n_directors
    print(f"Gender nodes: {n_genders}")

    # director -> gender edges
    dir_nodes_g = torch.tensor(
        [dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    gen_nodes   = torch.tensor(
        [gender2idx[g] + gender_offset for g in movies["director_gender"]], dtype=torch.long)

    dg_src = torch.cat([dir_nodes_g, gen_nodes])
    dg_dst = torch.cat([gen_nodes, dir_nodes_g])

    # ── 4. Region nodes ──
    regions       = ["western", "non-western", "unknown"]
    region2idx    = {r: i for i, r in enumerate(regions)}
    n_regions     = len(regions)
    region_offset = gender_offset + n_genders
    print(f"Region nodes: {n_regions}")

    # movie -> region edges
    reg_nodes = torch.tensor(
        [region2idx[r] + region_offset for r in movies["region"]], dtype=torch.long)

    mr_src = torch.cat([movie_nodes, reg_nodes])
    mr_dst = torch.cat([reg_nodes, movie_nodes])

    # ── 5. Genre nodes ──
    all_genres = set()
    for g in movies["genres"]:
        for genre in g.split("|"):
            all_genres.add(genre.strip())
    genres       = sorted(all_genres)
    genre2idx    = {g: i for i, g in enumerate(genres)}
    n_genres     = len(genres)
    genre_offset = region_offset + n_regions
    print(f"Genre nodes: {n_genres}")

    # movie -> genre edges (one movie can have multiple genres)
    mg_srcs, mg_dsts = [], []
    for _, row in movies.iterrows():
        m_node = int(row["movie_idx"]) + n_users
        for genre in row["genres"].split("|"):
            genre = genre.strip()
            if genre in genre2idx:
                g_node = genre2idx[genre] + genre_offset
                mg_srcs.append(m_node)
                mg_dsts.append(g_node)
                mg_srcs.append(g_node)
                mg_dsts.append(m_node)

    mg_src = torch.tensor(mg_srcs, dtype=torch.long)
    mg_dst = torch.tensor(mg_dsts, dtype=torch.long)

    # ── Combine all edges ──
    all_src = torch.cat([um_src, md_src, dg_src, mr_src, mg_src])
    all_dst = torch.cat([um_dst, md_dst, dg_dst, mr_dst, mg_dst])
    edge_index = torch.stack([all_src, all_dst], dim=0).to(device)

    n_total = n_users + n_movies + n_directors + n_genders + n_regions + n_genres

    node_map = {
        "n_users":        n_users,
        "n_movies":       n_movies,
        "n_directors":    n_directors,
        "n_genders":      n_genders,
        "n_regions":      n_regions,
        "n_genres":       n_genres,
        "n_total":        n_total,
        "user_offset":    0,
        "movie_offset":   n_users,
        "dir_offset":     dir_offset,
        "gender_offset":  gender_offset,
        "region_offset":  region_offset,
        "genre_offset":   genre_offset,
    }

    print(f"\nTotal nodes in KG: {n_total}")
    print(f"Total edges (both directions): {edge_index.shape[1]}")
    return edge_index, n_total, node_map


# ─── LIGHTGCN MODEL ───────────────────────────────────────────────────────────

class LightGCNModel(nn.Module):
    def __init__(self, n_total, n_users, n_movies, embedding_dim, num_layers):
        super().__init__()
        self.n_users   = n_users
        self.n_movies  = n_movies
        self.n_total   = n_total

        self.embedding = nn.Embedding(n_total, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.lgcn = LightGCN(n_total, embedding_dim, num_layers)
        self.lgcn.embedding = self.embedding

    def forward(self, edge_index):
        x = self.lgcn.get_embedding(edge_index)
        user_emb  = x[:self.n_users]
        movie_emb = x[self.n_users: self.n_users + self.n_movies]
        return user_emb, movie_emb

    def bpr_loss(self, user_emb, movie_emb, edge_index, n_users):
        # Only use user->movie edges for BPR sampling
        user_movie_mask = edge_index[0] < n_users
        um_edges = edge_index[:, user_movie_mask]

        src, pos, neg = structured_negative_sampling(
            um_edges,
            num_nodes=n_users + len(movie_emb)
        )

        src = src[src < n_users]
        pos = (pos[:len(src)] - n_users).clamp(0, len(movie_emb) - 1)
        neg = (neg[:len(src)] - n_users).clamp(0, len(movie_emb) - 1)

        u     = user_emb[src]
        pos_i = movie_emb[pos]
        neg_i = movie_emb[neg]

        loss = -F.logsigmoid((u * pos_i).sum(1) - (u * neg_i).sum(1)).mean()
        reg  = (u.norm(2).pow(2) + pos_i.norm(2).pow(2) + neg_i.norm(2).pow(2)) / len(src)
        return loss + 1e-4 * reg


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_epoch(model, edge_index, n_users, optimizer):
    model.train()
    optimizer.zero_grad()
    user_emb, movie_emb = model(edge_index)
    loss = model.bpr_loss(user_emb, movie_emb, edge_index, n_users)
    loss.backward()
    optimizer.step()
    # Free intermediate tensors immediately after each step
    torch.cuda.empty_cache()
    gc.collect()
    return loss.item()


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def get_recommendations(model, edge_index, train_df, n_users, n_movies):
    """
    Batched scoring to avoid OOM on large user sets.
    Scores SCORE_BATCH users at a time instead of all at once.
    """
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)
        # Move embeddings to CPU immediately to free GPU memory
        user_emb  = user_emb.cpu()
        movie_emb = movie_emb.cpu()

    torch.cuda.empty_cache()

    seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    recs = {}
    for start in range(0, n_users, SCORE_BATCH):
        end         = min(start + SCORE_BATCH, n_users)
        batch_emb   = user_emb[start:end]                          # [batch, dim]
        batch_scores = torch.matmul(batch_emb, movie_emb.T).numpy() # [batch, n_movies]

        for i, u in enumerate(range(start, end)):
            s = batch_scores[i].copy()
            for m in seen.get(u, set()):
                if m < len(s):
                    s[m] = -np.inf
            top = np.argpartition(s, -TOP_K)[-TOP_K:]
            top = top[np.argsort(s[top])[::-1]]
            recs[u] = top.tolist()

    return recs


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
        dcg       = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg      = sum(1 / np.log2(i + 2) for i in range(min(len(actual), TOP_K)))
        ndcg      = dcg / idcg if idcg > 0 else 0

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

    spd = np.mean(spd_list) if spd_list else 0.0
    eod = np.mean(eod_list) if eod_list else 0.0
    return spd, eod


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Load
    pos, movies, user2idx, movie2idx, n_users, n_movies = load_data()

    # Split (same logic as baseline for fair comparison)
    train_df, val_df, test_df = split_data(pos)

    # Build KG
    print("\nBuilding heterogeneous knowledge graph...")
    edge_index, n_total, node_map = build_kg(train_df, movies, n_users, n_movies)

    # Model
    model     = LightGCNModel(n_total, n_users, n_movies, EMBEDDING_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    print("\nTraining LightGCN on KG...")
    best_ndcg  = 0
    best_epoch = 0

    for epoch in tqdm(range(1, EPOCHS + 1)):
        loss = train_epoch(model, edge_index, n_users, optimizer)

        if epoch % 5 == 0:
            recs = get_recommendations(model, edge_index, train_df, n_users, n_movies)
            p, r, ndcg = precision_recall_ndcg(recs, val_df)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | P@10: {p:.4f} | R@10: {r:.4f} | NDCG@10: {ndcg:.4f}")

            if ndcg > best_ndcg:
                best_ndcg  = ndcg
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_kg.pt"))

    print(f"\nBest NDCG@10 on val: {best_ndcg:.4f} at epoch {best_epoch}")

    # Evaluate on test set with best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model_kg.pt")))
    recs = get_recommendations(model, edge_index, train_df, n_users, n_movies)
    p, r, ndcg = precision_recall_ndcg(recs, test_df)

    print(f"\n--- Test Set Results (KG) ---")
    print(f"Precision@10: {p:.4f}")
    print(f"Recall@10:    {r:.4f}")
    print(f"NDCG@10:      {ndcg:.4f}")

    # Fairness metrics
    spd_g, eod_g = compute_fairness_metrics(
        recs, test_df, movies, "director_gender", "female", "male")
    spd_r, eod_r = compute_fairness_metrics(
        recs, test_df, movies, "region", "non-western", "western")

    print(f"\n--- Fairness Metrics (KG) ---")
    print(f"Gender  SPD: {spd_g:.4f}")
    print(f"Gender  EOD: {eod_g:.4f}")
    print(f"Region  SPD: {spd_r:.4f}")
    print(f"Region  EOD: {eod_r:.4f}")

    # Load baseline results for comparison
    baseline_path = "outputs/baseline/baseline_results.json"
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"\n--- Comparison Table ---")
        print(f"{'Metric':<20} {'Baseline':>12} {'KG':>12} {'Change':>12}")
        print("-" * 58)
        metrics = [
            ("NDCG@10",      "ndcg_at_10",      ndcg),
            ("Precision@10", "precision_at_10",  p),
            ("Recall@10",    "recall_at_10",     r),
            ("Gender SPD",   "gender_spd",       spd_g),
            ("Gender EOD",   "gender_eod",       eod_g),
            ("Region SPD",   "region_spd",       spd_r),
            ("Region EOD",   "region_eod",       eod_r),
        ]
        for label, key, kg_val in metrics:
            base_val = baseline.get(key, 0)
            change   = kg_val - base_val
            print(f"{label:<20} {base_val:>12.4f} {kg_val:>12.4f} {change:>+12.4f}")

    # Save results
    results = {
        "model":           "LightGCN_heterogeneous_KG",
        "precision_at_10": round(p, 4),
        "recall_at_10":    round(r, 4),
        "ndcg_at_10":      round(ndcg, 4),
        "gender_spd":      round(spd_g, 4),
        "gender_eod":      round(eod_g, 4),
        "region_spd":      round(spd_r, 4),
        "region_eod":      round(eod_r, 4),
        "node_map":        node_map,
    }

    out_path = os.path.join(OUTPUT_DIR, "kg_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

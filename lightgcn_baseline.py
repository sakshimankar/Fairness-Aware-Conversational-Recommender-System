"""
FA-CRS Day 4-6: Bipartite Graph + LightGCN Baseline
-----------------------------------------------------
Steps:
1. Load enriched data from Day 1-3
2. Filter ratings (keep only ratings >= 4 as positive interactions)
3. Build bipartite user-movie graph
4. Train LightGCN on the graph
5. Evaluate: Precision@10, Recall@10, NDCG@10
6. Compute fairness metrics: SPD, EOD across gender and region groups
7. Save model + results

Requirements:
    pip install torch torch-geometric pandas numpy scikit-learn tqdm

NOTE: If torch-geometric fails to install on Windows, use Google Colab from this step.
Colab install commands are provided at the bottom of this file.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import structured_negative_sampling
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR       = "data"
OUTPUT_DIR     = "outputs/baseline"
EMBEDDING_DIM  = 32
NUM_LAYERS     = 2
LEARNING_RATE  = 1e-3
EPOCHS         = 30
BATCH_SIZE     = 4096
MIN_RATING     = 4        # threshold for "positive" interaction
TOP_K          = 10       # for evaluation metrics
RANDOM_SEED    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    ratings  = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies   = pd.read_csv(os.path.join(DATA_DIR, "movies_enriched.csv"))
    #users    = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))

    # Keep only positive interactions
    pos = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()
    print(f"Positive interactions (rating >= {MIN_RATING}): {len(pos)}")

    # Re-index users and movies to 0-based integer IDs
    user_ids  = sorted(pos["user_id"].unique())
    movie_ids = sorted(pos["movie_id"].unique())

    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}

    pos["user_idx"]  = pos["user_id"].map(user2idx)
    pos["movie_idx"] = pos["movie_id"].map(movie2idx)

    n_users  = len(user_ids)
    n_movies = len(movie_ids)
    print(f"Users: {n_users}, Movies: {n_movies}")

    # Attach enriched movie metadata
    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"])
    movies["movie_idx"] = movies["movie_idx"].astype(int)

    return pos, movies, user2idx, movie2idx, n_users, n_movies


# ─── TRAIN/VAL/TEST SPLIT ─────────────────────────────────────────────────────

def split_data(pos):
    """
    Split per user: 80% train, 10% val, 10% test.
    This avoids leaking future interactions into training.
    """
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


# ─── BUILD GRAPH ──────────────────────────────────────────────────────────────

def build_edge_index(train_df, n_users):
    """
    Build bipartite edge index for PyG.
    Movie node indices are offset by n_users so they don't overlap with user nodes.
    Shape: [2, num_edges * 2] (both directions for message passing)
    """
    user_idx  = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    movie_idx = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)

    # Both directions: user->movie and movie->user
    edge_index = torch.stack([
        torch.cat([user_idx, movie_idx]),
        torch.cat([movie_idx, user_idx])
    ], dim=0)

    return edge_index.to(device)


# ─── LIGHTGCN MODEL ───────────────────────────────────────────────────────────

class LightGCNModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, num_layers):
        super().__init__()
        self.n_users   = n_users
        self.n_movies  = n_movies
        self.num_nodes = n_users + n_movies
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Use PyG LightGCN with correct signature: (num_nodes, embedding_dim, num_layers)
        self.lgcn = LightGCN(self.num_nodes, embedding_dim, num_layers)
        # Override its embedding with ours so we control initialisation
        self.lgcn.embedding = self.embedding

    def forward(self, edge_index):
        # PyG LightGCN.forward returns final embeddings directly
        x = self.lgcn.get_embedding(edge_index)
        user_emb  = x[:self.n_users]
        movie_emb = x[self.n_users:]
        return user_emb, movie_emb

    def bpr_loss(self, user_emb, movie_emb, edge_index, n_users):
        """
        Bayesian Personalised Ranking loss.
        For each positive (user, movie) pair, sample a negative movie and
        push positive score above negative score.
        """
        src, pos, neg = structured_negative_sampling(
            edge_index[:, edge_index[0] < n_users],  # only user->movie edges
            num_nodes=n_users + len(movie_emb)
        )

        src  = src[src < n_users]
        pos  = pos[:len(src)] - n_users
        neg  = neg[:len(src)] - n_users

        # Clamp to valid range
        pos = pos.clamp(0, len(movie_emb) - 1)
        neg = neg.clamp(0, len(movie_emb) - 1)

        u   = user_emb[src]
        pos_i = movie_emb[pos]
        neg_i = movie_emb[neg]

        pos_score = (u * pos_i).sum(dim=1)
        neg_score = (u * neg_i).sum(dim=1)

        loss = -F.logsigmoid(pos_score - neg_score).mean()

        # L2 regularisation
        reg = (u.norm(2).pow(2) + pos_i.norm(2).pow(2) + neg_i.norm(2).pow(2)) / len(src)
        return loss + 1e-4 * reg


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train(model, edge_index, n_users, optimizer):
    model.train()
    optimizer.zero_grad()
    user_emb, movie_emb = model(edge_index)
    loss = model.bpr_loss(user_emb, movie_emb, edge_index, n_users)
    loss.backward()
    optimizer.step()
    return loss.item()


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def get_recommendations(model, edge_index, train_df, n_users, n_movies, top_k):
    """
    For every user, score all movies, exclude already-seen ones, return top-k.
    Returns dict: user_idx -> list of top-k movie_idx
    """
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)

    # Movies seen per user in training
    seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    SCORE_BATCH = 1000
    recs = {}
    for start in range(0, n_users, SCORE_BATCH):
        end = min(start + SCORE_BATCH, n_users)
        batch_scores = torch.matmul(user_emb[start:end], movie_emb.T).cpu().numpy()
        for i, u in enumerate(range(start, end)):
            seen_u = seen.get(u, set())
            s = batch_scores[i].copy()
            s[list(seen_u)] = -np.inf
            top = np.argpartition(s, -top_k)[-top_k:]
            top = top[np.argsort(s[top])[::-1]]
            recs[u] = top.tolist()

    return recs


def precision_recall_ndcg(recs, ground_truth_df, top_k):
    """
    Compute Precision@K, Recall@K, NDCG@K.
    ground_truth_df: dataframe with user_idx, movie_idx columns (val or test set)
    """
    gt = ground_truth_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []

    for u, rec_list in recs.items():
        if u not in gt:
            continue
        actual = gt[u]
        hits = [1 if m in actual else 0 for m in rec_list[:top_k]]

        precision = sum(hits) / top_k
        recall    = sum(hits) / len(actual) if actual else 0

        # NDCG
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), top_k)))
        ndcg = dcg / idcg if idcg > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


# ─── FAIRNESS METRICS ─────────────────────────────────────────────────────────

def compute_fairness_metrics(recs, test_df, movies, attribute_col, group_a, group_b):
    """
    Compute SPD and EOD for a binary sensitive attribute.

    SPD (Statistical Parity Difference):
        P(recommended | group_a) - P(recommended | group_b)

    EOD (Equal Opportunity Difference):
        P(recommended | relevant, group_a) - P(recommended | relevant, group_b)

    attribute_col: column in movies df ('director_gender' or 'region')
    group_a: e.g. 'female' or 'non-western'  (the disadvantaged group)
    group_b: e.g. 'male'   or 'western'
    """
    # Map movie_idx -> group
    movie_group = movies.set_index("movie_idx")[attribute_col].to_dict()

    gt = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    spd_list, eod_list = [], []

    for u, rec_list in recs.items():
        rec_set = set(rec_list)

        # Classify recommended movies
        rec_a = sum(1 for m in rec_set if movie_group.get(m) == group_a)
        rec_b = sum(1 for m in rec_set if movie_group.get(m) == group_b)
        total_rec = rec_a + rec_b
        if total_rec == 0:
            continue

        p_rec_a = rec_a / total_rec
        p_rec_b = rec_b / total_rec
        spd_list.append(p_rec_a - p_rec_b)

        # EOD: among relevant (ground truth) movies
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

    # Split
    train_df, val_df, test_df = split_data(pos)

    # Graph
    edge_index = build_edge_index(train_df, n_users)

    # Model
    model = LightGCNModel(n_users, n_movies, EMBEDDING_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    print("\nTraining LightGCN...")
    best_ndcg = 0
    best_epoch = 0

    for epoch in tqdm(range(1, EPOCHS + 1)):
        loss = train(model, edge_index, n_users, optimizer)

        if epoch % 5 == 0:
            recs = get_recommendations(model, edge_index, train_df, n_users, n_movies, TOP_K)
            p, r, ndcg = precision_recall_ndcg(recs, val_df, TOP_K)
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | P@10: {p:.4f} | R@10: {r:.4f} | NDCG@10: {ndcg:.4f}")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

    print(f"\nBest NDCG@10 on val: {best_ndcg:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
    recs = get_recommendations(model, edge_index, train_df, n_users, n_movies, TOP_K)
    p, r, ndcg = precision_recall_ndcg(recs, test_df, TOP_K)

    print(f"\n--- Test Set Results ---")
    print(f"Precision@10: {p:.4f}")
    print(f"Recall@10:    {r:.4f}")
    print(f"NDCG@10:      {ndcg:.4f}")

    # Fairness metrics
    spd_g, eod_g = compute_fairness_metrics(
        recs, test_df, movies,
        attribute_col="director_gender",
        group_a="female", group_b="male"
    )
    spd_r, eod_r = compute_fairness_metrics(
        recs, test_df, movies,
        attribute_col="region",
        group_a="non-western", group_b="western"
    )

    print(f"\n--- Fairness Metrics (Baseline) ---")
    print(f"Gender  SPD: {spd_g:.4f}  (negative = female underrepresented)")
    print(f"Gender  EOD: {eod_g:.4f}")
    print(f"Region  SPD: {spd_r:.4f}  (negative = non-western underrepresented)")
    print(f"Region  EOD: {eod_r:.4f}")

    # Save results
    results = {
        "model": "LightGCN_bipartite_baseline",
        "precision_at_10": round(p, 4),
        "recall_at_10":    round(r, 4),
        "ndcg_at_10":      round(ndcg, 4),
        "gender_spd":      round(spd_g, 4),
        "gender_eod":      round(eod_g, 4),
        "region_spd":      round(spd_r, 4),
        "region_eod":      round(eod_r, 4),
    }

    with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/baseline_results.json")


if __name__ == "__main__":
    main()


# ─── COLAB INSTALL COMMANDS (if Windows install fails) ────────────────────────
"""
Run these in a Colab cell before running this script:

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
!pip install torch-geometric
!pip install pandas numpy scikit-learn tqdm
"""

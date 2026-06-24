"""
FA-CRS Gradio Demo
------------------
Interactive demo for the fairness-aware movie recommender.

Pick a user, drag the gender / region fairness sliders, and watch the
FA*IR-reranked Top-10 update next to the plain relevance-only Top-10.

Run from the project root (same folder as fair_rerank.py), after you've
already run heterogeneous_kg.py at least once so outputs/kg/best_model_kg.pt
exists:

    pip install gradio
    python gradio_app.py

Requirements: same as fair_rerank.py, plus gradio
    pip install torch torch-geometric pandas numpy gradio tqdm
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gradio as gr
from torch_geometric.nn import LightGCN
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────

DATA_DIR      = "data"
KG_MODEL_PATH = "outputs/kg/best_model_kg.pt"
EMBEDDING_DIM = 64
NUM_LAYERS    = 3
MIN_RATING    = 4
TOP_K         = 10
CANDIDATE_K   = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── DATA / GRAPH / MODEL (identical logic to heterogeneous_kg.py / fair_rerank.py) ──

def load_data():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies_enriched.csv"))

    pos = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()

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

    return pos, movies, n_users, n_movies


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

    u_idx  = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m_idx  = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)
    um_src = torch.cat([u_idx, m_idx])
    um_dst = torch.cat([m_idx, u_idx])

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


# ─── FA*IR RERANKING (identical to fair_rerank.py) ───────────────────────────

def fair_rerank(candidates, movie_attr, protected_val, p, k=TOP_K):
    protected   = [(m, s) for m, s in candidates if movie_attr.get(m) == protected_val]
    unprotected = [(m, s) for m, s in candidates if movie_attr.get(m) != protected_val]

    result, result_flags = [], []
    p_ptr = u_ptr = 0

    for pos in range(k):
        n_protected = sum(1 for f in result_flags if f)
        needed = math.ceil(p * (pos + 1))

        if n_protected < needed and p_ptr < len(protected):
            result.append(protected[p_ptr][0]); result_flags.append(True); p_ptr += 1
        else:
            take_protected = (
                p_ptr < len(protected) and
                (u_ptr >= len(unprotected) or protected[p_ptr][1] >= unprotected[u_ptr][1])
            )
            if take_protected:
                result.append(protected[p_ptr][0]); result_flags.append(False); p_ptr += 1
            elif u_ptr < len(unprotected):
                result.append(unprotected[u_ptr][0]); result_flags.append(False); u_ptr += 1

        if len(result) == k:
            break

    return result, result_flags


def rerank_user(cands, movie_gender, movie_region, p_gender, p_region):
    reranked_gender, gender_flags = fair_rerank(cands, movie_gender, "female", p_gender)

    reranked_scores = {m: s for m, s in cands}
    reranked_cands  = [(m, reranked_scores.get(m, -np.inf)) for m in reranked_gender]
    seen_set = set(reranked_gender)
    for m, s in cands:
        if m not in seen_set:
            reranked_cands.append((m, s))

    reranked_region, region_flags = fair_rerank(reranked_cands, movie_region, "non-western", p_region)

    combined_flags = []
    for i, m in enumerate(reranked_region):
        if i < len(region_flags) and region_flags[i]:
            combined_flags.append("region")
        elif i < len(gender_flags) and gender_flags[i]:
            combined_flags.append("gender")
        else:
            combined_flags.append("relevance")

    return reranked_region, combined_flags


def precision_recall_ndcg(recs, ground_truth_df):
    gt = ground_truth_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    precisions, recalls, ndcgs = [], [], []
    for u, rec_list in recs.items():
        if u not in gt:
            continue
        actual = gt[u]
        hits = [1 if m in actual else 0 for m in rec_list[:TOP_K]]
        precisions.append(sum(hits) / TOP_K)
        recalls.append(sum(hits) / len(actual) if actual else 0)
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), TOP_K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return (float(np.mean(precisions)) if precisions else 0.0,
            float(np.mean(recalls)) if recalls else 0.0,
            float(np.mean(ndcgs)) if ndcgs else 0.0)


def compute_fairness_metrics(recs, test_df, movie_group_map, group_a, group_b):
    gt = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    spd_list, eod_list = [], []
    for u, rec_list in recs.items():
        rec_set = set(rec_list)
        rec_a = sum(1 for m in rec_set if movie_group_map.get(m) == group_a)
        rec_b = sum(1 for m in rec_set if movie_group_map.get(m) == group_b)
        total = rec_a + rec_b
        if total == 0:
            continue
        spd_list.append(rec_a / total - rec_b / total)
        relevant = gt.get(u, set())
        rel_a = sum(1 for m in relevant if movie_group_map.get(m) == group_a)
        rel_b = sum(1 for m in relevant if movie_group_map.get(m) == group_b)
        hit_a = sum(1 for m in relevant if m in rec_set and movie_group_map.get(m) == group_a)
        hit_b = sum(1 for m in relevant if m in rec_set and movie_group_map.get(m) == group_b)
        tpr_a = hit_a / rel_a if rel_a > 0 else None
        tpr_b = hit_b / rel_b if rel_b > 0 else None
        if tpr_a is not None and tpr_b is not None:
            eod_list.append(tpr_a - tpr_b)
    return (float(np.mean(spd_list)) if spd_list else 0.0,
            float(np.mean(eod_list)) if eod_list else 0.0)


# ─── STARTUP: load data + model once, cache per-user candidate pools ─────────

READY = False
LOAD_ERROR = ""
n_users = n_movies = 0
movies_indexed = None
movie_gender = movie_region = {}
test_df = None
ALL_CANDIDATES = {}

try:
    print("Loading data...")
    pos, movies, n_users, n_movies = load_data()
    train_df, val_df, test_df = split_data(pos)

    print("Rebuilding knowledge graph...")
    edge_index, n_total = build_kg(train_df, movies, n_users, n_movies)

    print("Loading trained KG model weights...")
    model = LightGCNModel(n_total, n_users, n_movies, EMBEDDING_DIM, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(KG_MODEL_PATH, map_location=device))
    model.eval()

    movies_indexed = movies.set_index("movie_idx")
    movie_gender   = movies_indexed["director_gender"].to_dict()
    movie_region   = movies_indexed["region"].to_dict()
    seen_by_user   = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    female_movies     = set(m for m, g in movie_gender.items() if g == "female")
    nonwestern_movies = set(m for m, r in movie_region.items() if r == "non-western")

    print("Scoring all users (one forward pass)...")
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)
    all_scores = torch.matmul(user_emb, movie_emb.T).cpu().numpy()

    def build_candidates(u):
        s = all_scores[u].copy()
        seen_u = seen_by_user.get(u, set())
        for m in seen_u:
            if m < len(s):
                s[m] = -np.inf
        k = min(CANDIDATE_K, len(s))
        top = np.argpartition(s, -k)[-k:]
        top = top[np.argsort(s[top])[::-1]]
        pool = set(top.tolist())
        for m in sorted([m for m in female_movies if m not in seen_u and m < len(s)],
                         key=lambda m: s[m], reverse=True)[:10]:
            pool.add(m)
        for m in sorted([m for m in nonwestern_movies if m not in seen_u and m < len(s)],
                         key=lambda m: s[m], reverse=True)[:10]:
            pool.add(m)
        pool_list = sorted(pool, key=lambda m: s[m] if s[m] > -1e8 else -1e9, reverse=True)
        return [(int(m), float(s[m])) for m in pool_list]

    print("Building candidate pools for every user (one-time)...")
    ALL_CANDIDATES = {u: build_candidates(u) for u in tqdm(range(n_users))}

    print(f"Ready. {n_users} users, {n_movies} movies.")
    READY = True

except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}"
    print(f"Could not load model/data: {LOAD_ERROR}")


# ─── DISPLAY HELPERS ──────────────────────────────────────────────────────────

FLAG_LABELS = {
    "relevance": "Relevance",
    "gender":    "Gender diversity pick",
    "region":    "Region diversity pick",
}


def rec_table(rec_list, flags):
    rows = []
    for rank, (m, flag) in enumerate(zip(rec_list, flags), 1):
        if movies_indexed is not None and m in movies_indexed.index:
            row = movies_indexed.loc[m]
            title    = row.get("title", f"Movie {m}")
            genres   = str(row.get("genres", "")).replace("|", ", ")
            director = row.get("director", "Unknown")
            gender   = row.get("director_gender", "unknown")
            region   = row.get("region", "unknown")
        else:
            title, genres, director, gender, region = f"Movie {m}", "", "Unknown", "unknown", "unknown"
        rows.append({
            "#": rank,
            "Title": title,
            "Genre": genres,
            "Director": director,
            "Director gender": gender,
            "Region": region,
            "Why recommended": FLAG_LABELS.get(flag, flag),
        })
    return pd.DataFrame(rows)


def group_counts(rec_list):
    female     = sum(1 for m in rec_list if movie_gender.get(m) == "female")
    nonwestern = sum(1 for m in rec_list if movie_region.get(m) == "non-western")
    return female, nonwestern


# ─── GRADIO CALLBACKS ──────────────────────────────────────────────────────────

def get_recommendations(user_id, p_gender, p_region):
    user_id = int(user_id)
    cands = ALL_CANDIDATES.get(user_id, [])

    baseline_list  = [m for m, s in cands[:TOP_K]]
    baseline_flags = ["relevance"] * len(baseline_list)

    fair_list, fair_flags = rerank_user(cands, movie_gender, movie_region, p_gender, p_region)

    baseline_table = rec_table(baseline_list, baseline_flags)
    fair_table     = rec_table(fair_list, fair_flags)

    b_female, b_nw = group_counts(baseline_list)
    f_female, f_nw = group_counts(fair_list)

    summary = pd.DataFrame([
        {"Metric": "Female-directed (of 10)", "Relevance only": b_female, "Fairness-reranked": f_female},
        {"Metric": "Non-western (of 10)",      "Relevance only": b_nw,     "Fairness-reranked": f_nw},
    ])

    return baseline_table, fair_table, summary


def compute_aggregate_metrics(p_gender, p_region):
    recs = {}
    for u in range(n_users):
        rec_list, _ = rerank_user(ALL_CANDIDATES.get(u, []), movie_gender, movie_region, p_gender, p_region)
        recs[u] = rec_list

    prec, rec, ndcg = precision_recall_ndcg(recs, test_df)
    spd_g, eod_g = compute_fairness_metrics(recs, test_df, movie_gender, "female", "male")
    spd_r, eod_r = compute_fairness_metrics(recs, test_df, movie_region, "non-western", "western")

    return pd.DataFrame([
        {"Metric": "NDCG@10",      "Value": round(ndcg, 4)},
        {"Metric": "Precision@10", "Value": round(prec, 4)},
        {"Metric": "Recall@10",    "Value": round(rec, 4)},
        {"Metric": "Gender SPD",   "Value": round(spd_g, 4)},
        {"Metric": "Gender EOD",   "Value": round(eod_g, 4)},
        {"Metric": "Region SPD",   "Value": round(spd_r, 4)},
        {"Metric": "Region EOD",   "Value": round(eod_r, 4)},
    ])


def random_user():
    return int(np.random.randint(0, max(n_users, 1)))


# ─── UI ────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="FA-CRS Live Demo") as demo:
        gr.Markdown("# FA-CRS: fairness-aware movie recommender — live demo")
        gr.Markdown(
            "Pick a user and drag the two fairness sliders. The left table is the "
            "knowledge-graph model's plain top-10 by relevance; the right table is "
            "the same candidate pool after FA*IR reranking."
        )

        with gr.Row():
            user_input = gr.Slider(0, max(n_users - 1, 1), value=0, step=1,
                                    label=f"User ID (0–{max(n_users - 1, 0)})")
            random_btn = gr.Button("Random user")

        with gr.Row():
            p_gender_input = gr.Slider(0.0, 0.6, value=0.3, step=0.05, label="Gender fairness target (p)")
            p_region_input = gr.Slider(0.0, 0.6, value=0.3, step=0.05, label="Region fairness target (p)")

        go_btn = gr.Button("Get recommendations", variant="primary")

        with gr.Row():
            baseline_out = gr.Dataframe(label="Relevance only (no fairness constraint)")
            fair_out     = gr.Dataframe(label="FA*IR reranked")

        summary_out = gr.Dataframe(label="Group representation in top 10")

        with gr.Accordion("Aggregate metrics across all users (slower, computed on demand)", open=False):
            agg_btn = gr.Button("Compute aggregate metrics for this p")
            agg_out = gr.Dataframe(label="NDCG / Precision / Recall / SPD / EOD at the current p")

        inputs  = [user_input, p_gender_input, p_region_input]
        outputs = [baseline_out, fair_out, summary_out]

        go_btn.click(get_recommendations, inputs=inputs, outputs=outputs)
        user_input.release(get_recommendations, inputs=inputs, outputs=outputs)
        p_gender_input.release(get_recommendations, inputs=inputs, outputs=outputs)
        p_region_input.release(get_recommendations, inputs=inputs, outputs=outputs)
        random_btn.click(random_user, outputs=user_input).then(
            get_recommendations, inputs=inputs, outputs=outputs)

        agg_btn.click(compute_aggregate_metrics, inputs=[p_gender_input, p_region_input], outputs=agg_out)

        demo.load(get_recommendations, inputs=inputs, outputs=outputs)

    return demo


def build_error_ui():
    with gr.Blocks(title="FA-CRS Live Demo") as demo:
        gr.Markdown("# FA-CRS live demo — setup needed")
        gr.Markdown(
            "This app couldn't load the trained model or data. From the project "
            "root, run these first, then restart this app:\n\n"
            "1. `python data_prep.py`\n"
            "2. `python lightgcn_baseline.py`\n"
            "3. `python heterogeneous_kg.py`\n\n"
            f"Error detail: `{LOAD_ERROR}`"
        )
    return demo


if __name__ == "__main__":
    app = build_ui() if READY else build_error_ui()
    app.launch()

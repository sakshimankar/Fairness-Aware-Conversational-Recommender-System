"""
FA-CRS Gradio Demo — with Ollama conversational layer
------------------------------------------------------
Run from the project root after heterogeneous_kg.py has been run:

    pip install gradio requests
    ollama pull llama3          # or whichever model you prefer
    python gradio_app.py

Fairness target p is fixed at 0.3 (the elbow of the FUT curve).
Ollama must be running locally: ollama serve
"""

import os, math, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import gradio as gr
from torch_geometric.nn import LightGCN
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR        = "data"
KG_MODEL_PATH   = "outputs/kg/best_model_kg.pt"
EMBEDDING_DIM   = 64
NUM_LAYERS      = 3
MIN_RATING      = 4
TOP_K           = 10
CANDIDATE_K     = 50
P_FAIRNESS      = 0.3          # fixed — elbow of FUT curve
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "llama3"     # change to llama3.2, mistral, etc. if preferred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── DATA / GRAPH / MODEL (same as before) ────────────────────────────────────

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
    genders      = ["female", "male", "unknown"]
    gender2idx   = {g: i for i, g in enumerate(genders)}
    n_genders    = 3
    gender_offset = dir_offset + n_directors
    regions      = ["western", "non-western", "unknown"]
    region2idx   = {r: i for i, r in enumerate(regions)}
    n_regions    = 3
    region_offset = gender_offset + n_genders
    all_genres = set()
    for g in movies["genres"]:
        for genre in g.split("|"):
            all_genres.add(genre.strip())
    genres      = sorted(all_genres)
    genre2idx   = {g: i for i, g in enumerate(genres)}
    n_genres    = len(genres)
    genre_offset = region_offset + n_regions

    u_idx  = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m_idx  = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)
    um_src = torch.cat([u_idx, m_idx]); um_dst = torch.cat([m_idx, u_idx])
    movie_nodes = torch.tensor(movies["movie_idx"].values + n_users, dtype=torch.long)
    dir_nodes   = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    md_src = torch.cat([movie_nodes, dir_nodes]); md_dst = torch.cat([dir_nodes, movie_nodes])
    dir_nodes_g = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    gen_nodes   = torch.tensor([gender2idx[g] + gender_offset for g in movies["director_gender"]], dtype=torch.long)
    dg_src = torch.cat([dir_nodes_g, gen_nodes]); dg_dst = torch.cat([gen_nodes, dir_nodes_g])
    reg_nodes = torch.tensor([region2idx[r] + region_offset for r in movies["region"]], dtype=torch.long)
    mr_src = torch.cat([movie_nodes, reg_nodes]); mr_dst = torch.cat([reg_nodes, movie_nodes])
    mg_srcs, mg_dsts = [], []
    for _, row in movies.iterrows():
        m_node = int(row["movie_idx"]) + n_users
        for genre in row["genres"].split("|"):
            genre = genre.strip()
            if genre in genre2idx:
                g_node = genre2idx[genre] + genre_offset
                mg_srcs.extend([m_node, g_node]); mg_dsts.extend([g_node, m_node])
    mg_src = torch.tensor(mg_srcs, dtype=torch.long); mg_dst = torch.tensor(mg_dsts, dtype=torch.long)
    all_src    = torch.cat([um_src, md_src, dg_src, mr_src, mg_src])
    all_dst    = torch.cat([um_dst, md_dst, dg_dst, mr_dst, mg_dst])
    edge_index = torch.stack([all_src, all_dst], dim=0).to(device)
    n_total    = n_users + n_movies + n_directors + n_genders + n_regions + n_genres
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


# ─── FA*IR RERANKING ──────────────────────────────────────────────────────────

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
            take_prot = (p_ptr < len(protected) and
                         (u_ptr >= len(unprotected) or protected[p_ptr][1] >= unprotected[u_ptr][1]))
            if take_prot:
                result.append(protected[p_ptr][0]); result_flags.append(False); p_ptr += 1
            elif u_ptr < len(unprotected):
                result.append(unprotected[u_ptr][0]); result_flags.append(False); u_ptr += 1
        if len(result) == k:
            break
    return result, result_flags


def rerank_user(cands, excluded_genres=None, include_genres=None):
    filtered = cands

    # Genre inclusion: keep only movies that match at least one requested genre.
    # Operates on the full candidate pool so every scored unseen movie is eligible,
    # not just the default top-50. FA*IR still runs on whatever subset remains.
    if include_genres:
        include_lower = [g.lower() for g in include_genres]
        filtered = [(m, s) for m, s in cands
                    if m in movies_indexed.index and
                    any(ig in movies_indexed.loc[m, "genres"].lower()
                        for ig in include_lower)]

    # Genre exclusion (can combine with inclusion).
    if excluded_genres:
        excluded_lower = [g.lower() for g in excluded_genres]
        filtered = [(m, s) for m, s in filtered
                    if not any(eg in movies_indexed.loc[m, "genres"].lower()
                               for eg in excluded_lower
                               if m in movies_indexed.index)]

    reranked_gender, gender_flags = fair_rerank(filtered, movie_gender, "female", P_FAIRNESS)
    reranked_scores = {m: s for m, s in filtered}
    reranked_cands  = [(m, reranked_scores.get(m, -np.inf)) for m in reranked_gender]
    seen_set = set(reranked_gender)
    for m, s in filtered:
        if m not in seen_set:
            reranked_cands.append((m, s))
    reranked_region, region_flags = fair_rerank(reranked_cands, movie_region, "non-western", P_FAIRNESS)
    combined_flags = []
    for i, m in enumerate(reranked_region):
        if i < len(region_flags) and region_flags[i]:
            combined_flags.append("region")
        elif i < len(gender_flags) and gender_flags[i]:
            combined_flags.append("gender")
        else:
            combined_flags.append("relevance")
    return reranked_region, combined_flags


# ─── OLLAMA ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the assistant for a Fairness-Aware Conversational Recommender System (FA-CRS) for movies.
Your job is to understand what the user wants and respond helpfully.

You handle four kinds of requests:

1. RECOMMEND — the user wants movies of a specific genre or type (e.g. "show me thrillers", "give me 10 horror films", "recommend some comedies").
   The system will fetch movies from the user's personalised model score AND apply FA*IR fairness reranking (p=0.3) to ensure gender and production country diversity.
   Respond with JSON only:
   {"intent": "recommend", "include_genres": ["Thriller"], "reason": "one friendly sentence telling the user what you're showing them and that fairness reranking is applied"}

2. FILTER — the user wants to remove certain genres from all recommendations (e.g. "no more action movies", "hide horror").
   Respond with JSON only:
   {"intent": "filter", "exclude_genres": ["Action"], "reason": "one sentence plain-English explanation"}

3. EXPLAIN — the user wants to know why a specific movie was recommended.
   Respond with JSON only:
   {"intent": "explain", "movie_title": "exact title from context"}

4. QUESTION — the user is asking something conversational (what is SPD? how does this work? why fairness? what is p=0.3?).
   Respond with JSON only:
   {"intent": "question", "answer": "your answer in 2-3 sentences, plain English, no jargon"}

Always return valid JSON. No markdown, no preamble. Pick the closest intent.
Key distinction: RECOMMEND = user wants a genre-focused list; FILTER = user wants to permanently hide a genre from all results."""

EXPLAIN_PROMPT = """You are explaining a movie recommendation to a user.

Movie: {title}
Genres: {genres}
Director: {director} ({gender}-directed, {region} production)
Why it appeared: {flag}

The user asked: {question}

Respond in 2-3 friendly sentences. Explain why this movie was recommended given their viewing history and the fairness goal of surfacing {flag_detail}. Be specific about the movie, not generic."""


def ollama_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_ollama(messages):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3}
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        return json.dumps({"intent": "question", "answer": f"Ollama error: {e}"})


def parse_intent(raw):
    """Parse Ollama's JSON response, fallback gracefully."""
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {"intent": "question", "answer": raw}


def build_watch_history_summary(user_idx):
    """Summarise what this user has already watched (for context to Ollama)."""
    seen = train_seen.get(user_idx, set())
    if not seen:
        return "no recorded watch history"
    rows = []
    for m in list(seen)[:10]:
        if m in movies_indexed.index:
            rows.append(movies_indexed.loc[m, "title"])
    return ", ".join(rows[:8]) + ("..." if len(rows) == 8 else "")


def generate_explanation(movie_idx, flag, user_question):
    """Ask Ollama to explain why a specific movie was recommended."""
    if movie_idx not in movies_indexed.index:
        return "I don't have details on that film."
    row = movies_indexed.loc[movie_idx]
    flag_detail = {
        "gender":    "films by female directors",
        "region":    "non-western productions",
        "relevance": "movies that match your taste profile",
    }.get(flag, "relevant movies")
    prompt = EXPLAIN_PROMPT.format(
        title=row.get("title", "Unknown"),
        genres=str(row.get("genres", "")).replace("|", ", "),
        director=row.get("director", "Unknown"),
        gender=row.get("director_gender", "unknown"),
        region=row.get("region", "unknown"),
        flag=flag,
        question=user_question,
        flag_detail=flag_detail,
    )
    raw = call_ollama([{"role": "user", "content": prompt}])
    return raw


# ─── STARTUP ─────────────────────────────────────────────────────────────────

READY = False
LOAD_ERROR = ""
n_users = n_movies = 0
movies_indexed = None
movie_gender = movie_region = {}
test_df = train_seen = {}
ALL_CANDIDATES = {}

try:
    print("Loading data...")
    pos, movies, n_users, n_movies = load_data()
    train_df, val_df, test_df = split_data(pos)
    train_seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    print("Rebuilding knowledge graph...")
    edge_index, n_total = build_kg(train_df, movies, n_users, n_movies)

    print("Loading trained KG model weights...")
    model = LightGCNModel(n_total, n_users, n_movies, EMBEDDING_DIM, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(KG_MODEL_PATH, map_location=device))
    model.eval()

    movies_indexed = movies.set_index("movie_idx")
    movie_gender   = movies_indexed["director_gender"].to_dict()
    movie_region   = movies_indexed["region"].to_dict()

    female_movies     = set(m for m, g in movie_gender.items() if g == "female")
    nonwestern_movies = set(m for m, r in movie_region.items() if r == "non-western")

    print("Scoring all users (one forward pass)...")
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)
    all_scores = torch.matmul(user_emb, movie_emb.T).cpu().numpy()

    def build_candidates(u):
        s = all_scores[u].copy()
        seen_u = train_seen.get(u, set())
        for m in seen_u:
            if m < len(s): s[m] = -np.inf
        k = min(CANDIDATE_K, len(s))
        top = np.argpartition(s, -k)[-k:]
        top = top[np.argsort(s[top])[::-1]]
        pool = set(top.tolist())
        for m in sorted([m for m in female_movies if m not in seen_u and m < len(s)],
                         key=lambda m: s[m], reverse=True)[:10]: pool.add(m)
        for m in sorted([m for m in nonwestern_movies if m not in seen_u and m < len(s)],
                         key=lambda m: s[m], reverse=True)[:10]: pool.add(m)
        pool_list = sorted(pool, key=lambda m: s[m] if s[m] > -1e8 else -1e9, reverse=True)
        return [(int(m), float(s[m])) for m in pool_list]

    print("Building candidate pools...")
    ALL_CANDIDATES = {u: build_candidates(u) for u in tqdm(range(n_users))}
    print(f"Ready. {n_users} users, {n_movies} movies. Ollama: {'available' if ollama_available() else 'not running'}")
    READY = True

except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}"
    print(f"Could not load: {LOAD_ERROR}")


# ─── DISPLAY HELPERS ─────────────────────────────────────────────────────────

FLAG_LABELS = {
    "relevance": "Relevance match",
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
            "#": rank, "Title": title, "Genre": genres,
            "Director": director, "Dir. gender": gender,
            "Region": region, "Why": FLAG_LABELS.get(flag, flag),
        })
    return pd.DataFrame(rows)


def get_recommendations(user_id, excluded_genres=None, include_genres=None):
    user_id = int(user_id)
    # Use the FULL candidate pool (all scored unseen movies) so genre filters
    # have the widest possible pool to draw from before FA*IR runs.
    cands = ALL_CANDIDATES.get(user_id, [])
    baseline_list  = [m for m, s in cands[:TOP_K]]
    fair_list, fair_flags = rerank_user(cands, excluded_genres, include_genres)
    return (rec_table(baseline_list, ["relevance"] * len(baseline_list)),
            rec_table(fair_list, fair_flags),
            fair_list, fair_flags)


# ─── CONVERSATIONAL HANDLER ──────────────────────────────────────────────────

def chat(user_message, history, user_id, current_fair_list, current_fair_flags, excluded_genres_state, include_genres_state):
    """
    Main chat callback. Returns updated history + possibly updated rec tables.

    include_genres_state: set when user asks for a specific genre ("show me thrillers").
                          Filters the KG candidate pool to that genre before FA*IR runs,
                          so results are both personalised and fairness-reranked.
    excluded_genres_state: genres permanently hidden from all results.
    """
    user_id = int(user_id)

    if not ollama_available():
        reply = ("Ollama isn't running. Start it with `ollama serve` in a terminal, "
                 "then refresh the page. In the meantime you can still browse the recommendation tables.")
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        return history, *get_recommendations(user_id, excluded_genres_state, include_genres_state)[:2], excluded_genres_state, include_genres_state

    # Build context about current recommendations for Ollama
    context_lines = ["Current FA*IR recommendations for this user:"]
    if include_genres_state:
        context_lines.append(f"  (currently filtered to genres: {', '.join(include_genres_state)})")
    if excluded_genres_state:
        context_lines.append(f"  (currently excluding genres: {', '.join(excluded_genres_state)})")
    for i, (m, flag) in enumerate(zip(current_fair_list[:TOP_K], current_fair_flags[:TOP_K]), 1):
        if movies_indexed is not None and m in movies_indexed.index:
            row = movies_indexed.loc[m]
            context_lines.append(
                f"  {i}. {row.get('title','?')} — {row.get('genres','').replace('|',', ')} "
                f"({row.get('director_gender','?')}-directed, {row.get('region','?')}) [{flag}]")

    context = "\n".join(context_lines)
    watch_summary = build_watch_history_summary(user_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nContext:\n{context}\nUser watch history sample: {watch_summary}"},
        {"role": "user",   "content": user_message}
    ]

    raw = call_ollama(messages)
    parsed = parse_intent(raw)
    intent = parsed.get("intent", "question")

    if intent == "recommend":
        # The user wants movies of a specific genre.
        # 1. Filter the full KG candidate pool to that genre (personalised scores).
        # 2. Run FA*IR reranking at p=0.3 on the filtered subset (fairness enforced).
        # The table now shows genre-specific results that are BOTH personalised
        # (ranked by the KG model's dot-product score for this user) AND fair
        # (at least p=0.3 female-directed and p=0.3 non-western within that genre).
        new_include = parsed.get("include_genres", [])
        reason      = parsed.get("reason", f"Showing {', '.join(new_include)} films with FA*IR fairness applied.")
        _, fair_table, new_fair_list, new_fair_flags = get_recommendations(
            user_id, excluded_genres_state, new_include)
        baseline_table, _, _, _ = get_recommendations(user_id)
        n_found = len(new_fair_list)
        if n_found == 0:
            reply = (f"I couldn't find any {', '.join(new_include)} films in your unrated candidate pool. "
                     "Try a different genre or say 'reset filters' to start over.")
            new_include = include_genres_state  # keep existing state unchanged
        else:
            reply = f"{reason} ({n_found} result{'s' if n_found != 1 else ''} found, fairness reranked at p={P_FAIRNESS}). Say 'show all genres' to clear."
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        return history, baseline_table, fair_table, excluded_genres_state, new_include

    elif intent == "filter":
        new_excluded = excluded_genres_state + parsed.get("exclude_genres", [])
        new_excluded = list(set(new_excluded))
        _, fair_table, new_fair_list, new_fair_flags = get_recommendations(
            user_id, new_excluded, include_genres_state)
        baseline_table, _, _, _ = get_recommendations(user_id)
        reason = parsed.get("reason", "Filtering applied.")
        excluded_str = ", ".join(new_excluded) if new_excluded else "none"
        reply = f"{reason} Currently excluding: {excluded_str}. Say 'reset filters' to clear."
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        return history, baseline_table, fair_table, new_excluded, include_genres_state

    elif intent == "explain":
        title_query = parsed.get("movie_title", "").lower()
        matched_idx = None
        for m in current_fair_list[:TOP_K]:
            if m in movies_indexed.index:
                t = str(movies_indexed.loc[m, "title"]).lower()
                if title_query in t or t in title_query:
                    matched_idx = m
                    break
        if matched_idx is None:
            reply = "I couldn't find that film in the current recommendations. Try asking about one of the titles shown."
        else:
            pos_in_list = current_fair_list.index(matched_idx)
            flag = current_fair_flags[pos_in_list] if pos_in_list < len(current_fair_flags) else "relevance"
            reply = generate_explanation(matched_idx, flag, user_message)
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        baseline_table, fair_table = get_recommendations(user_id, excluded_genres_state, include_genres_state)[:2]
        return history, baseline_table, fair_table, excluded_genres_state, include_genres_state

    else:
        answer = parsed.get("answer", raw)
        reset_msg = user_message.lower()
        if ("reset" in reset_msg or "clear" in reset_msg) and ("filter" in reset_msg or "genre" in reset_msg or "all" in reset_msg):
            answer = "Filters cleared. Showing your full personalised recommendations with FA*IR fairness applied."
            excluded_genres_state = []
            include_genres_state  = []
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
        baseline_table, fair_table = get_recommendations(user_id, excluded_genres_state, include_genres_state)[:2]
        return history, baseline_table, fair_table, excluded_genres_state, include_genres_state


# ─── UI ──────────────────────────────────────────────────────────────────────

def on_user_change(user_id):
    # Switching user resets genre filters — each user's candidate pool is different.
    baseline_table, fair_table, fair_list, fair_flags = get_recommendations(int(user_id))
    return baseline_table, fair_table, fair_list, fair_flags, [], []


def build_ui():
    with gr.Blocks(title="FA-CRS Live Demo") as demo:
        gr.Markdown("# FA-CRS — fairness-aware movie recommender")
        gr.Markdown(
            f"Fairness target fixed at **p = {P_FAIRNESS}** "
            "(elbow of the FUT curve — best fairness gain per accuracy cost). "
            "Use the chat to ask questions, filter genres, or explain any recommendation."
        )

        user_input = gr.Slider(0, max(n_users - 1, 1), value=0, step=1,
                               label=f"User ID (0–{max(n_users - 1, 0)})")

        with gr.Row():
            baseline_out = gr.Dataframe(label="Relevance only")
            fair_out     = gr.Dataframe(label=f"FA*IR reranked (p={P_FAIRNESS})")

        gr.Markdown("### Chat with the recommender")
        gr.Markdown(
            "Try: *'Give me 10 thriller movies'* · "
            "*'Why is the first film recommended?'* · "
            "*'No more action movies'* · "
            "*'What does SPD mean?'* · "
            "*'Show all genres'*"
        )

        chatbot = gr.Chatbot(height=340)
        with gr.Row():
            chat_input = gr.Textbox(placeholder="Ask about a recommendation or filter by genre...",
                                    show_label=False, scale=5)
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # Hidden state
        fair_list_state   = gr.State([])
        fair_flags_state  = gr.State([])
        excluded_state    = gr.State([])   # genres permanently hidden
        include_state     = gr.State([])   # genre the user asked to see (recommend intent)

        # Load initial recommendations
        def initial_recs(user_id):
            bt, ft, fl, ff = get_recommendations(int(user_id))
            return bt, ft, fl, ff, [], []

        demo.load(initial_recs, inputs=[user_input],
                  outputs=[baseline_out, fair_out, fair_list_state, fair_flags_state, excluded_state, include_state])

        user_input.release(on_user_change, inputs=[user_input],
                           outputs=[baseline_out, fair_out, fair_list_state, fair_flags_state, excluded_state, include_state])

        chat_inputs  = [chat_input, chatbot, user_input, fair_list_state, fair_flags_state, excluded_state, include_state]
        chat_outputs = [chatbot, baseline_out, fair_out, excluded_state, include_state]

        send_btn.click(chat, inputs=chat_inputs, outputs=chat_outputs).then(
            lambda: "", outputs=chat_input)
        chat_input.submit(chat, inputs=chat_inputs, outputs=chat_outputs).then(
            lambda: "", outputs=chat_input)

    return demo


def build_error_ui():
    with gr.Blocks(title="FA-CRS") as demo:
        gr.Markdown("# Setup needed")
        gr.Markdown(
            "Run the pipeline first:\n\n"
            "1. `python data_prep.py`\n"
            "2. `python lightgcn_baseline.py`\n"
            "3. `python heterogeneous_kg.py`\n\n"
            f"Error: `{LOAD_ERROR}`"
        )
    return demo


if __name__ == "__main__":
    app = build_ui() if READY else build_error_ui()
    app.launch()

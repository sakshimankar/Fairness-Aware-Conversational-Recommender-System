"""
CoT Reranking — Stage 3 of FA-CRS pipeline
LightGCN scores → FA*IR (fairness) → CoT (preference alignment) → final list + reasoning
"""

import os, json, math, time, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN
from tqdm import tqdm

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
KG_DIR        = "outputs/kg"
OUTPUT_DIR    = "outputs/cot"
EMBED_DIM     = 64
N_LAYERS      = 3
MIN_RATING    = 4
TOP_K         = 10
COT_USERS     = 200   # subset to run CoT on (full set is slow with LLM)
COT_MIN_PROT  = 0.25  # soft fairness floor: CoT can't drop below this protected proportion
LLM_MODEL     = "claude-sonnet-4-6"
API_DELAY     = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    movies  = pd.read_csv(f"{DATA_DIR}/movies_enriched.csv")
    pos     = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()
    user2idx  = {u: i for i, u in enumerate(sorted(pos["user_id"].unique()))}
    movie2idx = {m: i for i, m in enumerate(sorted(pos["movie_id"].unique()))}
    pos["user_idx"]  = pos["user_id"].map(user2idx)
    pos["movie_idx"] = pos["movie_id"].map(movie2idx)
    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    for col in ["director", "director_gender", "region", "genres", "title"]:
        movies[col] = movies.get(col, pd.Series(dtype=str)).fillna("unknown")
    return pos, movies, len(user2idx), len(movie2idx)


def split_data(pos):
    train, val, test = [], [], []
    for _, g in pos.groupby("user_idx"):
        items = g["movie_idx"].tolist()
        uid   = g["user_idx"].iloc[0]
        if len(items) < 3:
            train += [(uid, m) for m in items]; continue
        nv, nt = max(1, int(.1*len(items))), max(1, int(.1*len(items)))
        train += [(uid, m) for m in items[:-(nv+nt)]]
        val   += [(uid, m) for m in items[-(nv+nt):-nt]]
        test  += [(uid, m) for m in items[-nt:]]
    mk = lambda r: pd.DataFrame(r, columns=["user_idx","movie_idx"])
    return mk(train), mk(val), mk(test)


# ── MODEL (mirrors heterogeneous_kg.py) ───────────────────────────────────────
def build_kg(train_df, movies, n_users, n_movies):
    # node layout: users | movies | directors | genders | regions | genres
    dirs = sorted(movies["director"].unique());  dir2i = {d:i for i,d in enumerate(dirs)}
    gens = ["female","male","unknown"];           gen2i = {g:i for i,g in enumerate(gens)}
    regs = ["western","non-western","unknown"];  reg2i = {r:i for i,r in enumerate(regs)}
    all_g = sorted({g.strip() for gs in movies["genres"] for g in gs.split("|")})
    gnr2i = {g:i for i,g in enumerate(all_g)}

    do, go, ro, gno = n_users+n_movies, n_users+n_movies+len(dirs), 0, 0
    go = do + len(dirs); ro = go + len(gens); gno = ro + len(regs)

    u  = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m  = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)
    mn = torch.tensor(movies["movie_idx"].values + n_users, dtype=torch.long)
    dn = torch.tensor([dir2i[d]+do for d in movies["director"]], dtype=torch.long)
    gn = torch.tensor([gen2i[g]+go for g in movies["director_gender"]], dtype=torch.long)
    rn = torch.tensor([reg2i[r]+ro for r in movies["region"]], dtype=torch.long)

    mg_s, mg_d = [], []
    for _, row in movies.iterrows():
        mn_ = int(row["movie_idx"]) + n_users
        for genre in row["genres"].split("|"):
            genre = genre.strip()
            if genre in gnr2i:
                gnode = gnr2i[genre] + gno
                mg_s += [mn_, gnode]; mg_d += [gnode, mn_]

    def bi(a, b): return torch.cat([a,b]), torch.cat([b,a])
    es, ed = zip(*[bi(x,y) for x,y in [(u,m),(mn,dn),(dn,gn),(mn,rn),
                                         (torch.tensor(mg_s,dtype=torch.long),
                                          torch.tensor(mg_d,dtype=torch.long))]])
    edge_index = torch.stack([torch.cat(list(es)), torch.cat(list(ed))]).to(device)
    n_total = gno + len(all_g)
    return edge_index, n_total


class KGModel(nn.Module):
    def __init__(self, n_total, n_users, n_movies):
        super().__init__()
        self.n_users = n_users; self.n_movies = n_movies
        self.emb  = nn.Embedding(n_total, EMBED_DIM)
        self.lgcn = LightGCN(n_total, EMBED_DIM, N_LAYERS)
        self.lgcn.embedding = self.emb

    def forward(self, ei):
        x = self.lgcn.get_embedding(ei)
        return x[:self.n_users], x[self.n_users:self.n_users+self.n_movies]


# ── CANDIDATE SCORING ─────────────────────────────────────────────────────────
def get_candidates(model, edge_index, train_df, n_users, n_movies, movie_gender, movie_region):
    model.eval()
    with torch.no_grad():
        ue, me = model(edge_index)
    scores = torch.matmul(ue, me.T).cpu().numpy()
    seen   = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    f_pool = {m for m,g in movie_gender.items() if g == "female"}
    nw_pool= {m for m,r in movie_region.items() if r == "non-western"}

    cands = {}
    for u in range(n_users):
        s = scores[u].copy()
        for m in seen.get(u, set()):
            if m < len(s): s[m] = -np.inf
        top = set(np.argpartition(s,-50)[-50:].tolist())
        # inject top protected movies so FA*IR always has something to promote
        for pool in [f_pool, nw_pool]:
            for m in sorted([m for m in pool if m not in seen.get(u,set()) and m<len(s)],
                            key=lambda m: s[m], reverse=True)[:10]:
                top.add(m)
        cands[u] = [(int(m), float(s[m])) for m in sorted(top, key=lambda m: s[m] if s[m]>-1e8 else -1e9, reverse=True)]
    return cands


# ── FA*IR ─────────────────────────────────────────────────────────────────────
def fair_rerank(cands, attr, protected, p, k=TOP_K):
    prot = [(m,s) for m,s in cands if attr.get(m)==protected]
    unprot=[(m,s) for m,s in cands if attr.get(m)!=protected]
    result, flags, pi, ui = [], [], 0, 0
    for pos in range(k):
        need = math.ceil(p*(pos+1)) - sum(flags)
        if need > 0 and pi < len(prot):
            result.append(prot[pi][0]); flags.append(True); pi += 1
        elif pi < len(prot) and (ui>=len(unprot) or prot[pi][1]>=unprot[ui][1]):
            result.append(prot[pi][0]); flags.append(False); pi += 1
        elif ui < len(unprot):
            result.append(unprot[ui][0]); flags.append(False); ui += 1
        if len(result)==k: break
    return result, flags


def run_fair(cands, movie_gender, movie_region, p=0.3):
    recs, flags = {}, {}
    for u, c in cands.items():
        rg, gf = fair_rerank(c, movie_gender, "female", p)
        sc = {m:s for m,s in c}
        rc = [(m, sc.get(m,-np.inf)) for m in rg] + [(m,s) for m,s in c if m not in set(rg)]
        rr, rf = fair_rerank(rc, movie_region, "non-western", p)
        recs[u]  = rr
        flags[u] = ["region" if i<len(rf) and rf[i] else "gender" if i<len(gf) and gf[i] else "relevance"
                    for i in range(len(rr))]
    return recs, flags


# ── PREFERENCE PROFILE ────────────────────────────────────────────────────────
def infer_profile(user_idx, train_df, movies):
    seen = movies[movies["movie_idx"].isin(train_df[train_df["user_idx"]==user_idx]["movie_idx"])]
    gc = {}
    for gs in seen["genres"].fillna(""):
        for g in gs.split("|"):
            gc[g.strip()] = gc.get(g.strip(), 0) + 1
    liked = sorted(gc, key=gc.get, reverse=True)[:3]
    years = [int(m.group(1)) for t in seen.get("title", pd.Series()).fillna("")
             for m in [re.search(r"\((\d{4})\)", t)] if m]
    era   = f"{(int(np.median(years))//10)*10}s" if years else None
    div   = ((seen["region"]=="non-western").sum() + (seen["director_gender"]=="female").sum()) / max(len(seen),1)
    return {"liked": liked, "era": era, "diversity": "high" if div>.15 else "medium" if div>.05 else "low"}


# ── CoT ───────────────────────────────────────────────────────────────────────
def rule_based_cot(cand_dicts, profile, k=TOP_K):
    liked = set(profile["liked"])
    for c in cand_dicts:
        match = len({g.strip() for g in c["genres"].split("|")} & liked)
        bonus = 0.15*(c["director_gender"]=="female") + 0.15*(c["region"]=="non-western") \
                if profile["diversity"] != "low" else 0
        c["_cot_score"] = c["score"] + 0.1*match + bonus
    ranked = sorted(cand_dicts, key=lambda c: c["_cot_score"], reverse=True)
    # enforce soft fairness floor
    prot_min = math.ceil(COT_MIN_PROT * k)
    prot_count = sum(1 for c in ranked[:k] if c["director_gender"]=="female" or c["region"]=="non-western")
    if prot_count < prot_min:
        prot_extra = [c for c in ranked[k:] if c["director_gender"]=="female" or c["region"]=="non-western"]
        swap_idxs  = [i for i,c in enumerate(ranked[:k]) if c["director_gender"]!="female" and c["region"]!="western"]
        for si, pc in zip(swap_idxs, prot_extra):
            if prot_count >= prot_min: break
            ranked[si] = pc; prot_count += 1
    steps = [f"#{i+1} {c['title']} — {'diversity pick (' + c['director_gender'] + ', ' + c['region'] + ')' if c['director_gender']=='female' or c['region']=='non-western' else 'relevance pick'}, genres: {c['genres']}"
             for i, c in enumerate(ranked[:k])]
    return [c["movie_idx"] for c in ranked[:k]], steps


def llm_cot(cand_dicts, profile, client, k=TOP_K):
    lines = [f"{i+1}. {c['title']} | {c['genres']} | {c['director']} ({c['director_gender']}) | {c['region']} | score:{c['score']:.3f}"
             + (" [fairness-promoted]" if c["fairness_flag"]!="relevance" else "")
             for i, c in enumerate(cand_dicts)]
    prompt = f"""Re-rank these {k} movies for a user who likes {', '.join(profile['liked'])} (diversity appetite: {profile['diversity']}).

Candidates:
{chr(10).join(lines)}

Think step by step, then output:
REASONING: <brief per-item reasoning>
FINAL_RANKING:
1. <title>
...{k}. <title>

Constraint: at least {math.ceil(COT_MIN_PROT*k)} items must be female-directed OR non-western."""

    try:
        resp = client.messages.create(model=LLM_MODEL, max_tokens=600,
                                      messages=[{"role":"user","content":prompt}])
        text = resp.content[0].text
    except Exception as e:
        print(f"  LLM error: {e}, falling back to rule-based")
        ids, steps = rule_based_cot(cand_dicts, profile, k)
        return ids, "\n".join(steps)

    title2idx = {c["title"]: c["movie_idx"] for c in cand_dicts}
    final_ids = []
    if "FINAL_RANKING:" in text:
        for line in text.split("FINAL_RANKING:")[-1].strip().split("\n"):
            line = re.sub(r"^\d+\.\s*","", line.strip())
            for title, idx in title2idx.items():
                if title.lower() in line.lower() and idx not in final_ids:
                    final_ids.append(idx); break
            if len(final_ids)==k: break
    # pad with FA*IR order if parse incomplete
    for c in cand_dicts:
        if len(final_ids)==k: break
        if c["movie_idx"] not in final_ids: final_ids.append(c["movie_idx"])
    return final_ids[:k], text


# ── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate(recs, test_df, movies, k=TOP_K):
    gt = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    P, R, N = [], [], []
    for u, rl in recs.items():
        if u not in gt: continue
        hits = [1 if m in gt[u] else 0 for m in rl[:k]]
        P.append(sum(hits)/k)
        R.append(sum(hits)/len(gt[u]) if gt[u] else 0)
        dcg  = sum(h/np.log2(i+2) for i,h in enumerate(hits))
        idcg = sum(1/np.log2(i+2) for i in range(min(len(gt[u]),k)))
        N.append(dcg/idcg if idcg else 0)

    def fairness(attr_col, ga, gb):
        mg = movies.set_index("movie_idx")[attr_col].to_dict()
        spd, eod = [], []
        for u, rl in recs.items():
            rs = set(rl); ra = sum(1 for m in rs if mg.get(m)==ga); rb = sum(1 for m in rs if mg.get(m)==gb)
            if ra+rb: spd.append(ra/(ra+rb) - rb/(ra+rb))
            rel = gt.get(u,set())
            ha = sum(1 for m in rel if m in rs and mg.get(m)==ga)
            hb = sum(1 for m in rel if m in rs and mg.get(m)==gb)
            rla = sum(1 for m in rel if mg.get(m)==ga); rlb = sum(1 for m in rel if mg.get(m)==gb)
            if rla and rlb: eod.append(ha/rla - hb/rlb)
        # exposure gap: position-discounted group presence
        ea = [sum(1/np.log2(i+2) for i,m in enumerate(recs.get(u,[])) if mg.get(m)==ga) for u in recs]
        eb = [sum(1/np.log2(i+2) for i,m in enumerate(recs.get(u,[])) if mg.get(m)==gb) for u in recs]
        return (np.mean(spd) if spd else 0, np.mean(eod) if eod else 0,
                round(np.mean(ea)-np.mean(eb), 4))

    spd_g, eod_g, dexp_g = fairness("director_gender", "female", "male")
    spd_r, eod_r, dexp_r = fairness("region", "non-western", "western")
    return {"ndcg_at_10": round(np.mean(N),4), "precision_at_10": round(np.mean(P),4),
            "recall_at_10": round(np.mean(R),4), "gender_spd": round(spd_g,4),
            "gender_eod": round(eod_g,4), "region_spd": round(spd_r,4),
            "region_eod": round(eod_r,4), "gender_dexp": dexp_g, "region_dexp": dexp_r}


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_llm = bool(api_key and ANTHROPIC_AVAILABLE)
    client  = anthropic.Anthropic(api_key=api_key) if use_llm else None
    print(f"Backend: {'LLM (' + LLM_MODEL + ')' if use_llm else 'rule-based (set ANTHROPIC_API_KEY to use LLM)'}")

    pos, movies, n_users, n_movies = load_data()
    train_df, _, test_df = split_data(pos)
    mi = movies.set_index("movie_idx")
    movie_gender = mi["director_gender"].to_dict()
    movie_region = mi["region"].to_dict()

    print("Building KG + loading model...")
    edge_index, n_total = build_kg(train_df, movies, n_users, n_movies)
    model = KGModel(n_total, n_users, n_movies).to(device)
    model.load_state_dict(torch.load(f"{KG_DIR}/best_model_kg.pt", map_location=device))

    print("Scoring candidates...")
    cands = get_candidates(model, edge_index, train_df, n_users, n_movies, movie_gender, movie_region)

    print("FA*IR reranking...")
    fair_recs, fair_flags = run_fair(cands, movie_gender, movie_region)

    print(f"CoT reranking on {COT_USERS} users...")
    cot_recs, cot_reasons = {}, {}
    for u in tqdm(list(fair_recs.keys())[:COT_USERS]):
        profile    = infer_profile(u, train_df, movies)
        cand_dicts = [{"movie_idx": m, "title": mi.loc[m,"title"] if m in mi.index else f"Movie{m}",
                       "genres": mi.loc[m,"genres"] if m in mi.index else "", "director": mi.loc[m,"director"] if m in mi.index else "",
                       "director_gender": mi.loc[m,"director_gender"] if m in mi.index else "unknown",
                       "region": mi.loc[m,"region"] if m in mi.index else "unknown",
                       "score": next((s for mm,s in cands[u] if mm==m), 0.0),
                       "fairness_flag": fair_flags[u][i] if i<len(fair_flags[u]) else "relevance"}
                      for i, m in enumerate(fair_recs[u]) if m in mi.index]
        if not cand_dicts:
            cot_recs[u] = fair_recs[u]; cot_reasons[u] = "no metadata"; continue
        if use_llm:
            ids, reason = llm_cot(cand_dicts, profile, client); time.sleep(API_DELAY)
        else:
            ids, steps  = rule_based_cot(cand_dicts, profile); reason = "\n".join(steps)
        cot_recs[u] = ids; cot_reasons[u] = reason

    # evaluate CoT and FA*IR on same user subset for fair delta
    subset_fair = {u: fair_recs[u] for u in cot_recs}
    m_fair = evaluate(subset_fair, test_df, movies)
    m_cot  = evaluate(cot_recs,   test_df, movies)

    # load upstream results for comparison table
    def load_j(p): return json.load(open(p)) if os.path.exists(p) else {}
    base = load_j("outputs/baseline/baseline_results.json")
    kg   = load_j("outputs/kg/kg_results.json")
    fair_res = load_j("outputs/fair/fair_results.json")
    fp   = next((r for r in fair_res.get("fut_curve",[]) if r.get("p")==0.3), m_fair)

    print(f"\n{'Metric':<22}{'Baseline':>10}{'KG':>10}{'FA*IR':>10}{'CoT':>10}{'ΔCoT':>8}")
    print("-"*62)
    keys = [("NDCG@10","ndcg_at_10"),("Precision@10","precision_at_10"),("Recall@10","recall_at_10"),
            ("Gender SPD","gender_spd"),("Gender EOD","gender_eod"),("Region SPD","region_spd"),("Region EOD","region_eod")]
    for label, k_ in keys:
        b,kg_,f,c = base.get(k_,0), kg.get(k_,0), fp.get(k_,m_fair.get(k_,0)), m_cot.get(k_,0)
        print(f"{label:<22}{b:>10.4f}{kg_:>10.4f}{f:>10.4f}{c:>10.4f}{c-f:>+8.4f}")
    print(f"\nExposure Gap  | Gender: FA*IR={m_fair['gender_dexp']:+.4f} CoT={m_cot['gender_dexp']:+.4f}"
          f" | Region: FA*IR={m_fair['region_dexp']:+.4f} CoT={m_cot['region_dexp']:+.4f}")

    json.dump({"model": f"CoT_{'llm' if use_llm else 'rule'}", **m_cot,
               "fair_metrics": m_fair, "n_users": len(cot_recs)},
              open(f"{OUTPUT_DIR}/cot_results.json","w"), indent=2)

    with open(f"{OUTPUT_DIR}/cot_examples.txt","w",encoding="utf-8") as f:
        for u in list(cot_recs.keys())[:5]:
            profile = infer_profile(u, train_df, movies)
            f.write(f"User {u} | likes: {profile['liked']} | diversity: {profile['diversity']}\n")
            f.write(cot_reasons.get(u,"") + "\n")
            for rank, m in enumerate(cot_recs[u], 1):
                if m in mi.index:
                    f.write(f"  {rank}. {mi.loc[m,'title']} [{mi.loc[m,'director_gender']}, {mi.loc[m,'region']}]\n")
            f.write("\n")

    print(f"\nSaved: {OUTPUT_DIR}/cot_results.json, cot_examples.txt")


if __name__ == "__main__":
    main()

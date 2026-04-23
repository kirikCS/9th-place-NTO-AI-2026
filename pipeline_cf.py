"""
CF Ensemble: ItemItemCF + UserUserCF + ALS + AuthorPop → Optuna-tuned RRF
==========================================================================
Lightweight retrieval approach: collaborative filtering models
blended via Optuna-optimized RRF weights.

Required:  pip install implicit optuna tqdm scipy pandas numpy
"""
import pandas as pd
import numpy as np
import math
import random
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path
import optuna
import scipy.sparse as sp
import warnings, sys, time

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
IS_BUILD_SUBMIT = True
METRIC_TO_OPTIMIZE = "RECALL@200"
TOP_K_RECS = 200
OPT_TRIALS_BLEND = 100
OPTUNA_USER_SAMPLE = 1000

DATA_DIR = os.environ.get('DATA_DIR', 'data')
CACHE_DIR = '.'
data_path = Path(DATA_DIR)
output_dir = Path(CACHE_DIR)

def load_validation_data():
    obs = pd.read_parquet(output_dir / "observable_interactions.parquet")
    hid = pd.read_parquet(output_dir / "hidden_positives.parquet")
    ed = pd.read_csv(data_path / "editions.csv", engine='python')
    return obs, hid, ed

def load_full_data():
    df = pd.read_csv(data_path / "interactions.csv")
    targets = pd.read_csv(data_path / "targets.csv")
    ed = pd.read_csv(data_path / "editions.csv", engine='python')
    return df, targets, ed

def calculate_metrics(recs, rel, k_ndcg=20, k_recall=200):
    ndcgs, recalls = [], []
    for u, true_items in rel.items():
        u_recs = recs.get(u, [])
        u_ndcg = u_recs[:k_ndcg]
        dcg = sum([1.0 / math.log2(i + 2) for i, r in enumerate(u_ndcg) if r in true_items])
        idcg = sum([1.0 / math.log2(i + 2) for i in range(min(len(true_items), k_ndcg))])
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
        u_recall = set(u_recs[:k_recall])
        hits = len(u_recall & true_items)
        recalls.append(hits / len(true_items) if len(true_items) > 0 else 0)
    return {"NDCG@20": np.mean(ndcgs) if ndcgs else 0, "RECALL@200": np.mean(recalls) if recalls else 0}

# --- MODELS ---

class ItemItemCF:
    def __init__(self, mode='cosine', incident_only=False, k_sims=500):
        self.mode = mode; self.incident_only = incident_only; self.k_sims = k_sims
        self.sims = defaultdict(lambda: defaultdict(float)); self.user_history = {}; self.it_to_auth = {}

    def fit(self, df, editions, gap_bounds=None):
        self.sims = defaultdict(lambda: defaultdict(float))
        pos = df[df['event_type'].isin([1, 2])].copy(); pos['event_ts'] = pd.to_datetime(pos['event_ts'])
        if self.incident_only and gap_bounds:
            g_start, g_end = pd.to_datetime(gap_bounds[0]), pd.to_datetime(gap_bounds[1])
            pos = pos[(pos['event_ts'] >= g_start) & (pos['event_ts'] < g_end)]
        pos = pos.drop_duplicates(['user_id', 'edition_id']).sort_values(['user_id', 'event_ts'])
        self.user_history = pos.groupby('user_id')[['edition_id', 'event_ts']].apply(lambda x: list(zip(x['edition_id'], x['event_ts']))).to_dict()
        item_pop = pos['edition_id'].value_counts(); self.it_to_auth = editions.set_index('edition_id')['author_id'].to_dict()

        for u, group in tqdm(pos.groupby('user_id'), desc=f"Fitting CF {self.mode}", leave=False):
            history = group['edition_id'].tolist()
            for i, item1 in enumerate(history):
                for j in range(i + 1, min(i + 16, len(history))):
                    item2 = history[j]; boost = 1.3 if self.it_to_auth.get(item1) == self.it_to_auth.get(item2) else 1.0
                    weight = (1.0 / (j - i)) * boost
                    self.sims[item1][item2] += weight; self.sims[item2][item1] += weight
        for item, neighbors in self.sims.items():
            if self.mode == 'cosine':
                norm = item_pop[item]
                for n_item in neighbors: neighbors[n_item] /= math.sqrt(norm * item_pop[n_item])
            self.sims[item] = dict(sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:self.k_sims])

    def recommend_scores(self, user_id, gap_bounds, k=500, gap_weight=3.0, author_boost=1.3):
        history = self.user_history.get(user_id, [])
        if not history: return {}
        g_start, g_end = pd.to_datetime(gap_bounds[0]), pd.to_datetime(gap_bounds[1])
        scores = defaultdict(float); seen = set([x[0] for x in history])
        for item, ts in history:
            w = gap_weight if g_start <= ts < g_end else 1.0 / (1.0 + min(abs((ts - g_start).total_seconds()), abs((ts - g_end).total_seconds())) / 86400.0)
            for sim_item, sim_val in self.sims.get(item, {}).items():
                if sim_item not in seen:
                    boost = author_boost if self.it_to_auth.get(item) == self.it_to_auth.get(sim_item) else 1.0
                    scores[sim_item] += sim_val * w * boost
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k])

    def recommend(self, user_id, gap_bounds, k=250, gap_weight=3.0, author_boost=1.3):
        scores = self.recommend_scores(user_id, gap_bounds, k=k, gap_weight=gap_weight, author_boost=author_boost)
        return list(scores.keys())

class UserUserCF:
    def __init__(self):
        self.user_to_items = {}; self.item_to_users = defaultdict(list)
    def fit(self, df):
        self.user_to_items = {}; self.item_to_users = defaultdict(list); pos = df[df['event_type'].isin([1, 2])]
        self.user_to_items = pos.groupby('user_id')['edition_id'].apply(set).to_dict()
        for u, items in self.user_to_items.items():
            for i in items: self.item_to_users[i].append(u)
    def recommend_scores(self, user_id, gap_items, k_neighbors=100, k=500):
        u_items = self.user_to_items.get(user_id, set())
        if not u_items: return {}
        user_scores = defaultdict(float)
        for item in u_items:
            others = self.item_to_users.get(item, []); w = 1.0 / math.log1p(len(others))
            for other_u in others: 
                if other_u != user_id: user_scores[other_u] += w
        top_sims = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:k_neighbors]
        item_scores = defaultdict(float)
        for other_u, sim_weight in top_sims:
            for item in self.user_to_items.get(other_u, set()):
                if item not in u_items and item in gap_items: item_scores[item] += sim_weight
        return dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k])

    def recommend(self, user_id, gap_items, k_neighbors=100, k=250):
        scores = self.recommend_scores(user_id, gap_items, k_neighbors=k_neighbors, k=k)
        return list(scores.keys())

def generate_author_pop(history, editions, target_users, n=200, items_per_auth=40):
    it_to_auth = editions.set_index('edition_id')['author_id'].to_dict(); auth_to_its = editions.groupby('author_id')['edition_id'].apply(list).to_dict()
    pop = history[history['event_type'].isin([1, 2])]['edition_id'].value_counts().to_dict(); u_items = history[history['event_type'].isin([1, 2])].groupby('user_id')['edition_id'].apply(set).to_dict()
    res = {}
    for u in target_users:
        seen = u_items.get(u, set()); auths = set([it_to_auth.get(eid) for eid in seen if it_to_auth.get(eid)]); cands = []
        for aid in auths: its = sorted([i for i in auth_to_its.get(aid, []) if i not in seen], key=lambda x: pop.get(x, 0), reverse=True)[:items_per_auth]; cands.extend(its)
        sorted_cands = sorted(list(set(cands)), key=lambda x: pop.get(x, 0), reverse=True)[:n]
        res[u] = {eid: pop.get(eid, 0) for eid in sorted_cands}
    return res

def generate_als(history, target_users, k=300, factors=256, iterations=20, regularization=0.007563586589490131, alpha=5.327799631653635):
    """ALS via implicit library. Returns dict {user_id: [edition_id, ...]}"""
    try:
        from implicit.als import AlternatingLeastSquares
        from implicit.nearest_neighbours import bm25_weight
    except ImportError:
        print("implicit not found!")
        return {u: [] for u in target_users}

    print(f"Building ALS model (factors={factors}, iterations={iterations}, alpha={alpha}, bm25=True)...")
    positives = history[history['event_type'].isin([1, 2])].copy()
    positives['user_id'] = positives['user_id'].astype('category')
    positives['edition_id'] = positives['edition_id'].astype('category')
    
    u_idx = positives['user_id'].cat.codes.values
    i_idx = positives['edition_id'].cat.codes.values
    user_cats = positives['user_id'].cat.categories
    item_cats = positives['edition_id'].cat.categories
    
    u_map = {u: i for i, u in enumerate(user_cats)}
    i_map_rev = dict(enumerate(item_cats))

    conf_values = np.where(positives['event_type'].values == 2, 2.0, 1.0).astype(np.float32)
    ui_mat = sp.csr_matrix(
        (conf_values * alpha, (u_idx, i_idx)),
        shape=(len(user_cats), len(item_cats))
    )
    ui_mat = bm25_weight(ui_mat, K1=1.2, B=0.75).tocsr()

    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations, regularization=regularization,
        random_state=42, num_threads=0, use_gpu=False
    )
    model.fit(ui_mat, show_progress=True)
    
    valid_target_users = [u for u in target_users if u in u_map]
    target_u_idxs = np.array([u_map[u] for u in valid_target_users], dtype=np.int32)
    
    ids, dists = model.recommend(target_u_idxs, ui_mat[target_u_idxs], N=k, filter_already_liked_items=True)
    
    res = {u: {} for u in target_users}
    for i, u in enumerate(valid_target_users):
        res[u] = {i_map_rev[idx]: dists[i][j] for j, idx in enumerate(ids[i])}
    return res

def blend_rrf_local(target_users, pools, user_seen, global_pop, weights, editions_df, rrf_k=60, k=TOP_K_RECS):
    eid_to_bid = editions_df.set_index('edition_id')['book_id'].to_dict(); hybrid = {}
    for u in target_users:
        scores = defaultdict(float); s_eids = user_seen.get(u, set())
        s_bids = set([eid_to_bid.get(e) for e in s_eids if eid_to_bid.get(e)]) # Seen book IDs
        for p_name, p_dict in pools.items():
            w = weights.get(p_name, 0.0)
            if w <= 1e-4: continue
            # p_dict.get(u, {}) is now a dict {edition_id: score}, sorted by score
            for rank, eid in enumerate(p_dict.get(u, {})):
                if eid not in s_eids:
                    bid = eid_to_bid.get(eid)
                    if bid not in s_bids: # FILTER SEEN BOOKS
                         scores[eid] += w / (rrf_k + rank + 1)
        res = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        if len(res) < k:
            cur_bids = set([eid_to_bid.get(e) for e in res if eid_to_bid.get(e)])
            for eid in global_pop:
                bid = eid_to_bid.get(eid)
                if eid not in s_eids and bid not in cur_bids: res.append(eid); cur_bids.add(bid)
                if len(res) >= k: break
        hybrid[u] = res[:k]
    return hybrid

if __name__ == "__main__":
    obs_v, hid_v, ed_v = load_validation_data(); v_gap = ("2025-10-01", "2025-11-01")
    rel_v = hid_v.groupby('user_id')['edition_id'].apply(set).to_dict(); target_all = list(rel_v.keys())
    seen_v = obs_v.groupby('user_id')['edition_id'].apply(set).to_dict()
    v_gap_items = set(obs_v[(pd.to_datetime(obs_v['event_ts']) >= pd.to_datetime(v_gap[0])) & (pd.to_datetime(obs_v['event_ts']) < pd.to_datetime(v_gap[1]))]['edition_id'].unique())

    print("\n--- STAGE 1: FITTING MODELS (FIXED PARAMS) ---")
    m_count = ItemItemCF(mode='count'); m_count.fit(obs_v, ed_v)
    m_cos = ItemItemCF(mode='cosine'); m_cos.fit(obs_v, ed_v)
    uucf = UserUserCF(); uucf.fit(obs_v)

    fixed_ii = {"gap_weight": 3.0, "author_boost": 1.3}

    print("\n--- STAGE 2: GENERATING POOLS ---")
    best_pools = {
        "cf_count": {u: m_count.recommend_scores(u, v_gap, **fixed_ii) for u in tqdm(target_all, desc="Pool Count")},
        "cf_cos": {u: m_cos.recommend_scores(u, v_gap, **fixed_ii) for u in tqdm(target_all, desc="Pool Cosine")},
        "author": generate_author_pop(obs_v, ed_v, target_all),
        "user_cf": {u: uucf.recommend_scores(u, v_gap_items) for u in tqdm(target_all, desc="Pool UserCF")},
        "als": generate_als(obs_v, target_all)
    }

    print("\n--- STANDALONE PERFORMANCE ---")
    for name, p_dict in best_pools.items():
        # calculate_metrics expects recs as {user: [ids]}
        recs_list = {u: list(it_scores.keys()) for u, it_scores in p_dict.items()}
        m = calculate_metrics(recs_list, rel_v); print(f"{name:<15} | NDCG@20: {m['NDCG@20']:.4f} | RECALL@200: {m['RECALL@200']:.4f}")

    print("\n--- STAGE 3: BLENDING OPTIMIZATION ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    g_pop_v = obs_v[obs_v['event_type'].isin([1, 2])]['edition_id'].value_counts().index.tolist()
    tracker_b = {"best": -1.0}; study_blend = optuna.create_study(direction="maximize")
    with tqdm(total=OPT_TRIALS_BLEND, desc="Opt Blending") as pbar:
        def obj_b(trial):
            w = {p: trial.suggest_float(f"w_{p}", 0, 5) for p in best_pools.keys()}
            rk = trial.suggest_int("rrf_k", 20, 150)
            h = blend_rrf_local(target_all, best_pools, seen_v, g_pop_v, w, ed_v, rrf_k=rk); m = calculate_metrics(h, rel_v); val = m[METRIC_TO_OPTIMIZE]; pbar.update(1)
            if val > tracker_b["best"]: tracker_b["best"] = val
            pbar.set_postfix({"best_ndcg": f"{tracker_b['best']:.4f}"}); return val
        study_blend.optimize(obj_b, n_trials=OPT_TRIALS_BLEND, n_jobs=1)

    params = study_blend.best_params
    best_weights = {k.replace('w_', ''): v for k, v in params.items() if k.startswith('w_')}
    best_k = params.get("rrf_k", 60)
    final_v_res = calculate_metrics(blend_rrf_local(target_all, best_pools, seen_v, g_pop_v, best_weights, ed_v, rrf_k=best_k), rel_v)
    print(f"\n[GOLDEN] FINAL VALIDATION NDCG@20: {final_v_res['NDCG@20']:.4f} | RECALL@200: {final_v_res['RECALL@200']:.4f} | K={best_k}")

    if IS_BUILD_SUBMIT:
        print("\n--- BUILDING SUBMISSION ---")
        f_df, f_targets, f_ed = load_full_data(); t_f = f_targets['user_id'].unique().tolist()
        f_gap_bounds = ("2025-10-01", "2025-11-01")
        m_count.fit(f_df, f_ed); m_cos.fit(f_df, f_ed); uucf.fit(f_df)
        f_g_items = set(f_df[(pd.to_datetime(f_df['event_ts']) >= pd.to_datetime(f_gap_bounds[0])) & (pd.to_datetime(f_df['event_ts']) < pd.to_datetime(f_gap_bounds[1]))]['edition_id'].unique())
        f_pools = {
            "cf_count": {u: m_count.recommend_scores(u, f_gap_bounds, **fixed_ii) for u in tqdm(t_f, desc="Full Count")},
            "cf_cos": {u: m_cos.recommend_scores(u, f_gap_bounds, **fixed_ii) for u in tqdm(t_f, desc="Full Cosine")},
            "author": generate_author_pop(f_df, f_ed, t_f),
            "user_cf": {u: uucf.recommend_scores(u, f_g_items) for u in tqdm(t_f, desc="Full UserCF")},
            "als": generate_als(f_df, t_f)
        }
        f_seen = f_df.groupby('user_id')['edition_id'].apply(set).to_dict(); f_pop = f_df[f_df['event_type'].isin([1, 2])]['edition_id'].value_counts().index.tolist()
        final_recs = blend_rrf_local(t_f, f_pools, f_seen, f_pop, best_weights, f_ed, rrf_k=best_k); sub = []
        for u, its in final_recs.items():
            for rank, eid in enumerate(its[:20], 1): sub.append({'user_id': u, 'edition_id': eid, 'rank': rank})
        pd.DataFrame(sub).to_csv("sub_cf.csv", index=False); print('Saved: sub_cf.csv')

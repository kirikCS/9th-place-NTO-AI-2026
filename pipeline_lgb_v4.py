"""
LTR v4: U2U + I2I + Surge + Bridge → LightGBM LambdaRank
=========================================================
- Candidate sources: U2U CF, I2I co-occurrence, Surge detection,
  Bridge patterns, Continuation, Author, EASE (enriched + incident), SVD
- ~90 features (CF scores, temporal, tabular, cross-features)
- LightGBM LambdaRank, 3 folds, early stopping
- Post-processing: book dedup, author cap (3), fallback padding

Required: pip install lightgbm scipy scikit-learn pandas numpy
"""
import os, sys, gc, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

np.random.seed(42)

 # CONFIG
DATA_DIR = os.environ.get('DATA_DIR', 'data')
INCIDENT_START = pd.Timestamp('2025-10-01')
INCIDENT_END   = pd.Timestamp('2025-11-01')
SEED = 42
MAX_CANDS = 500

FOLDS = [
    (pd.Timestamp('2025-10-01'), pd.Timestamp('2025-08-01'), pd.Timestamp('2025-09-01')),
    (pd.Timestamp('2025-10-01'), pd.Timestamp('2025-09-01'), pd.Timestamp('2025-10-01')),
    (pd.Timestamp('2025-08-01'), pd.Timestamp('2025-06-01'), pd.Timestamp('2025-07-01')),
]

def log(*a):
    sys.stdout.write(' '.join(map(str, a)) + '\n')
    sys.stdout.flush()

def now():
    return time.strftime('%H:%M:%S')

 # HELPERS
 
def simulate_hidden(inter_df, inc_start, inc_end, seed=42):
    p = inter_df[
        (inter_df['event_ts'] >= inc_start) & (inter_df['event_ts'] < inc_end) &
        (inter_df['event_type'].isin([1, 2]))
    ]
    pairs = p[['user_id', 'edition_id']].drop_duplicates()
    if len(pairs) == 0:
        return inter_df.copy(), set(), defaultdict(set), []
    rng = np.random.RandomState(seed)
    hi = rng.choice(len(pairs), max(1, int(len(pairs) * 0.2)), replace=False)
    hdf = pairs.iloc[hi].copy()
    hs = set(zip(hdf['user_id'], hdf['edition_id']))
    hbu = defaultdict(set)
    for u, e in hs:
        hbu[u].add(e)
    rm = p[p.apply(lambda r: (r['user_id'], r['edition_id']) in hs, axis=1)].index
    vis = inter_df[~inter_df.index.isin(rm)].copy()
    return vis, hs, hbu, sorted(hbu.keys())

def global_fallback(inter_df, inc_start, inc_end):
    p = inter_df[inter_df['event_type'].isin([1, 2])]
    c = Counter()
    c.update(p[(p['event_ts'] >= inc_start) & (p['event_ts'] < inc_end)]['edition_id'].value_counts().to_dict())
    c.update(p[p['event_ts'] >= inc_end]['edition_id'].value_counts().to_dict())
    return [eid for eid, _ in c.most_common(5000)]

def slice_by_end(df, end_ts):
    if 'event_ts' in df.columns:
        return df[df['event_ts'] < end_ts].copy()
    return df.copy()

def precompute_feats(editions, book_genres):
    item_author = dict(zip(editions['edition_id'], editions['author_id']))
    item_book = dict(zip(editions['edition_id'], editions['book_id']))
    item_publisher = dict(zip(editions['edition_id'], editions['publisher_id']))
    ed_lang = dict(zip(editions['edition_id'], editions['language_id']))
    ed_year = dict(zip(editions['edition_id'], editions['publication_year']))
    ed_age = dict(zip(editions['edition_id'], editions['age_restriction']))
    author_editions = defaultdict(set)
    for eid, aid in item_author.items():
        if aid != -1:
            author_editions[aid].add(eid)
    book_editions = defaultdict(set)
    for eid, bid in item_book.items():
        if bid != -1:
            book_editions[bid].add(eid)
    publisher_editions = defaultdict(set)
    for eid, pid in item_publisher.items():
        if pid != -1:
            publisher_editions[pid].add(eid)
    book_genre_map = book_genres.groupby('book_id')['genre_id'].apply(set).to_dict()
    item_genres = {}
    for eid, bid in item_book.items():
        item_genres[eid] = book_genre_map.get(bid, set())
    return {
        'item_author': item_author, 'item_book': item_book, 'item_publisher': item_publisher,
        'ed_lang': ed_lang, 'ed_year': ed_year, 'ed_age': ed_age,
        'author_editions': author_editions, 'book_editions': book_editions,
        'publisher_editions': publisher_editions, 'item_genres': item_genres
    }

 # CANDIDATE SOURCES
 
def fit_ease_pairs(pairs, top_items=5000, min_pop=5, lam=25.0):
    if len(pairs) == 0:
        return None
    vc = pairs['edition_id'].value_counts()
    keep = vc[vc >= min_pop].head(top_items).index.tolist()
    if len(keep) < 2:
        return None
    part = pairs[pairs['edition_id'].isin(set(keep))][['user_id', 'edition_id']].drop_duplicates().copy()
    user_ids = sorted(part['user_id'].unique())
    item_ids = keep
    u2i = {u: i for i, u in enumerate(user_ids)}
    e2i = {e: i for i, e in enumerate(item_ids)}
    rows = part['user_id'].map(u2i).values
    cols = part['edition_id'].map(e2i).values
    X = csr_matrix((np.ones(len(part), dtype=np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    d = np.diag(P).copy()
    B = -P / d[np.newaxis, :]
    np.fill_diagonal(B, 0.0)
    return {'X': X, 'u2i': u2i, 'item_ids': item_ids, 'e2i': e2i, 'B': B}

def ease_user_top(model, uid, seen_items, topn=300):
    if model is None or uid not in model['u2i']:
        return {}
    row = model['X'][model['u2i'][uid]].toarray().ravel()
    sc = row @ model['B']
    for eid in seen_items:
        j = model['e2i'].get(eid)
        if j is not None:
            sc[j] = -1e18
    k = min(topn, len(sc))
    if k <= 0:
        return {}
    idx = np.argpartition(-sc, k - 1)[:k]
    idx = idx[np.argsort(-sc[idx])]
    out = {}
    for j in idx:
        if sc[j] <= -1e17:
            continue
        out[model['item_ids'][j]] = float(sc[j])
    return out

def fit_knn_pairs(pairs, min_pop=3, top_items=6000, n_neighbors=50):
    if len(pairs) == 0:
        return None
    part = pairs[['user_id', 'edition_id']].drop_duplicates().copy()
    item_pop = part['edition_id'].value_counts()
    keep = item_pop[item_pop >= min_pop].head(top_items).index.tolist()
    if len(keep) < 2:
        return None
    part = part[part['edition_id'].isin(set(keep))]
    user_ids = sorted(part['user_id'].unique())
    item_ids = keep
    u2i = {u: i for i, u in enumerate(user_ids)}
    e2i = {e: i for i, e in enumerate(item_ids)}
    rows = part['user_id'].map(u2i).values
    cols = part['edition_id'].map(e2i).values
    X = csr_matrix((np.ones(len(part), dtype=np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    item_vecs = X.T.tocsr()
    k = min(n_neighbors, len(item_ids))
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(item_vecs)
    dists, idxs = nn.kneighbors(item_vecs)
    knn_nb = {}
    for i, eid in enumerate(item_ids):
        knn_nb[eid] = [(item_ids[idxs[i][j]], 1.0 - float(dists[i][j])) for j in range(k)]
    return {'X': X, 'u2i': u2i, 'item_ids': item_ids, 'e2i': e2i, 'knn_nb': knn_nb}

def knn_user_top(model, uid, topn=300):
    if model is None or uid not in model['u2i']:
        return {}
    row = model['X'][model['u2i'][uid]]
    seen_idx = row.indices
    seen_items = set(model['item_ids'][j] for j in seen_idx)
    sc = defaultdict(float)
    for j in seen_idx:
        eid = model['item_ids'][j]
        for ne, sim in model['knn_nb'].get(eid, []):
            if ne not in seen_items:
                sc[ne] += sim
    if not sc:
        return {}
    items = sorted(sc.items(), key=lambda x: -x[1])[:topn]
    return dict(items)

def author_user_top(obs_pairs, editions, target_uids, topn=300):
    item_author = dict(zip(editions['edition_id'], editions['author_id']))
    author_items = editions.groupby('author_id')['edition_id'].apply(list).to_dict()
    item_pop = obs_pairs['edition_id'].value_counts().to_dict()
    seen = obs_pairs.groupby('user_id')['edition_id'].apply(set).to_dict()
    out = {}
    for uid in target_uids:
        u_seen = seen.get(uid, set())
        auth_cnt = Counter(item_author.get(eid, -1) for eid in u_seen if item_author.get(eid, -1) != -1)
        sc = {}
        for aid, cnt in auth_cnt.items():
            base = cnt * cnt if cnt >= 3 else cnt
            for eid in author_items.get(aid, []):
                if eid in u_seen:
                    continue
                sc[eid] = max(sc.get(eid, -1e18), float(base * 100 + item_pop.get(eid, 0)))
        out[uid] = dict(sorted(sc.items(), key=lambda x: -x[1])[:topn])
    return out

def rrf_from_dicts(dicts, k=60):
    sc = defaultdict(float)
    for d in dicts:
        rank_items = sorted(d.items(), key=lambda x: -x[1])
        for rank, (eid, _) in enumerate(rank_items):
            sc[eid] += 1.0 / (k + rank + 1)
    return dict(sorted(sc.items(), key=lambda x: -x[1]))

def build_sources(observed_interactions, hack_interactions, target_uids, editions):
    obs_pairs = observed_interactions[observed_interactions['event_type'].isin([1, 2])][['user_id', 'edition_id']].drop_duplicates().copy()
    enr_pairs = hack_interactions[hack_interactions['event_type'].isin([1, 2])][['user_id', 'edition_id']].drop_duplicates().copy()
    obs_seen = obs_pairs.groupby('user_id')['edition_id'].apply(set).to_dict()
    ease_model = fit_ease_pairs(enr_pairs, top_items=5000, min_pop=5, lam=25.0)
    knn_model = fit_knn_pairs(obs_pairs, min_pop=3, top_items=6000, n_neighbors=50)
    author_top = author_user_top(obs_pairs, editions, target_uids, topn=300)
    curr_hack, curr_knn, curr_author, curr_rrf = {}, {}, {}, {}
    for uid in target_uids:
        seen_u = obs_seen.get(uid, set())
        de = ease_user_top(ease_model, uid, seen_u, topn=300)
        dk = knn_user_top(knn_model, uid, topn=300)
        da = author_top.get(uid, {})
        dr = rrf_from_dicts([de, dk, da], k=60)
        curr_hack[uid] = de
        curr_knn[uid] = dk
        curr_author[uid] = da
        curr_rrf[uid] = dr
    return {'enr_ease': curr_hack, 'enr_knn': curr_knn, 'enr_auth': curr_author, 'enr_rrf': curr_rrf}

 # MATRIX + PREPARE
 
def build_matrix(inter_df, inc_start, inc_end):
    all_users = sorted(inter_df['user_id'].unique())
    all_items = sorted(inter_df['edition_id'].unique())
    uid2idx = {u: i for i, u in enumerate(all_users)}
    iid2idx = {e: i for i, e in enumerate(all_items)}
    idx2iid = {i: e for e, i in iid2idx.items()}
    idx2uid = {i: u for u, i in uid2idx.items()}
    rows = inter_df['user_id'].map(uid2idx).values
    cols = inter_df['edition_id'].map(iid2idx).values
    evt = inter_df['event_type'].values
    rating = inter_df['rating'].fillna(0).values
    ts = inter_df['event_ts']
    ew = np.where(evt == 2,
        np.where(rating >= 8, 5.0, np.where(rating >= 6, 3.0, np.where(rating >= 1, 1.5, 3.0))),
        2.0).astype(np.float32)
    tw = np.ones(len(inter_df), dtype=np.float32)
    bm = ts < inc_start
    im = (ts >= inc_start) & (ts < inc_end)
    am = ts >= inc_end
    if bm.any():
        bt = ts[bm]
        ts_min = bt.min()
        rng = (inc_start - ts_min).total_seconds()
        if rng > 0:
            tw[bm] = (0.5 + 0.5 * (bt - ts_min).dt.total_seconds() / rng).astype(np.float32)
    tw[im] = 2.0
    tw[am] = 2.0
    mat = csr_matrix((ew * tw, (rows, cols)), shape=(len(all_users), len(all_items)), dtype=np.float32)
    return mat, uid2idx, iid2idx, idx2iid, idx2uid

def prepare_sources(inter_df, hack_inter_df, target_uids, feats, editions, inc_start, inc_end):
    t0 = time.time()
    pos = inter_df[inter_df['event_type'].isin([1, 2])].copy()
    before_pos = pos[pos['event_ts'] < inc_start].copy()
    incident_pos = pos[(pos['event_ts'] >= inc_start) & (pos['event_ts'] < inc_end)].copy()
    after_pos = pos[pos['event_ts'] >= inc_end].copy()
    seen = pos.groupby('user_id')['edition_id'].agg(set).to_dict()
    user_inc = incident_pos.groupby('user_id')['edition_id'].agg(set).to_dict()
    user_bef = before_pos.groupby('user_id')['edition_id'].agg(set).to_dict()
    user_aft = after_pos.groupby('user_id')['edition_id'].agg(set).to_dict()

    log(f'[{now()}] prepare sources...')
    b1 = build_sources(inter_df, hack_inter_df, target_uids, editions)

    log(f'[{now()}] prepare matrix...')
    mat, uid2idx, iid2idx, idx2iid, idx2uid = build_matrix(inter_df, inc_start, inc_end)

    log(f'[{now()}] prepare U2U neighbors...')
    target_idx = np.array([uid2idx[u] for u in target_uids if u in uid2idx])
    mat_norm = normalize(mat, norm='l2', axis=1)
    neighbors = {}
    for start in range(0, len(target_idx), 500):
        end = min(start + 500, len(target_idx))
        batch = target_idx[start:end]
        sim = mat_norm[batch].dot(mat_norm.T).toarray()
        for i, tidx in enumerate(batch):
            row = sim[i].copy()
            row[tidx] = -1
            topk = np.argpartition(row, -100)[-100:]
            topk = topk[np.argsort(-row[topk])]
            sims = row[topk]
            mask = sims > 0.01
            neighbors[tidx] = (topk[mask], sims[mask])
    del mat_norm; gc.collect()

    log(f'[{now()}] prepare SVD...')
    svd_rows = pos['user_id'].map(lambda x: uid2idx.get(x, -1)).values
    svd_cols = pos['edition_id'].map(lambda x: iid2idx.get(x, -1)).values
    svd_mask = (svd_rows >= 0) & (svd_cols >= 0)
    svd_w = np.where(pos['event_type'].values == 2, 3.0, 1.0).astype(np.float32)
    svd_mat = csr_matrix((svd_w[svd_mask], (svd_rows[svd_mask], svd_cols[svd_mask])),
        shape=(len(uid2idx), len(iid2idx)), dtype=np.float32)
    svd_mat.data = np.log1p(svd_mat.data)
    k = min(32, min(svd_mat.shape) - 2)
    U, sigma, Vt = svds(svd_mat.astype(np.float64), k=k)
    sqrts = np.sqrt(sigma)
    uf = (U * sqrts).astype(np.float32)
    itf = (Vt.T * sqrts).astype(np.float32)
    svd_scores = {}
    for uid in target_uids:
        if uid in uid2idx:
            svd_scores[uid] = uf[uid2idx[uid]]
    del svd_mat, U, sigma, Vt; gc.collect()

    log(f'[{now()}] prepare EASE...')
    ease_pop = pos['edition_id'].value_counts()
    ease_top_items = sorted(ease_pop.head(12000).index)
    ease_item_set = set(ease_top_items)
    ease_item_idx = {e: i for i, e in enumerate(ease_top_items)}
    ease_item_rev = {i: e for e, i in ease_item_idx.items()}
    ease_users = sorted(pos[pos['edition_id'].isin(ease_item_set)]['user_id'].unique())
    ease_user_idx = {u: i for i, u in enumerate(ease_users)}
    pos_ease = pos[pos['edition_id'].isin(ease_item_set) & pos['user_id'].isin(set(ease_users))]
    er = pos_ease['user_id'].map(ease_user_idx).values
    ec = pos_ease['edition_id'].map(ease_item_idx).values
    ew_ease = np.where(pos_ease['event_type'].values == 2, 3.0, 1.0).astype(np.float32)
    X_ease = csr_matrix((ew_ease, (er, ec)), shape=(len(ease_users), len(ease_top_items)))
    X_ease.data = np.log1p(X_ease.data).astype(np.float32)
    XtX = (X_ease.T @ X_ease).toarray().astype(np.float64)
    eye_n = np.eye(len(ease_top_items))

    ease_user_scores = {}
    ease_user_scores2 = {}
    for lam, store in [(150.0, ease_user_scores), (500.0, ease_user_scores2)]:
        P_inv = np.linalg.inv(XtX + lam * eye_n)
        diag_p = np.diag(P_inv).copy()
        B = (-P_inv / diag_p[np.newaxis, :]).astype(np.float32)
        np.fill_diagonal(B, 0)
        for uid in target_uids:
            if uid in ease_user_idx:
                x_row = X_ease[ease_user_idx[uid]].toarray().ravel()
                store[uid] = x_row @ B
        del P_inv, B
    del XtX, eye_n; gc.collect()

    log(f'[{now()}] prepare EASE incident...')
    inc_ease_items = sorted(incident_pos['edition_id'].value_counts().head(5000).index)
    inc_ease_idx = {e: i for i, e in enumerate(inc_ease_items)}
    inc_ease_users = sorted(incident_pos[incident_pos['edition_id'].isin(set(inc_ease_items))]['user_id'].unique())
    inc_ease_uidx = {u: i for i, u in enumerate(inc_ease_users)}
    pos_inc_ease = incident_pos[incident_pos['edition_id'].isin(set(inc_ease_items))]
    ir = pos_inc_ease['user_id'].map(inc_ease_uidx).values
    ic_e = pos_inc_ease['edition_id'].map(inc_ease_idx).values
    iw_e = np.where(pos_inc_ease['event_type'].values == 2, 3.0, 1.0).astype(np.float32)
    X_inc_ease = csr_matrix((iw_e, (ir, ic_e)), shape=(len(inc_ease_users), len(inc_ease_items)))
    X_inc_ease.data = np.log1p(X_inc_ease.data).astype(np.float32)
    XtX_inc = (X_inc_ease.T @ X_inc_ease).toarray().astype(np.float64)
    P_inv_inc = np.linalg.inv(XtX_inc + 100.0 * np.eye(len(inc_ease_items)))
    diag_inc = np.diag(P_inv_inc).copy()
    B_inc_ease = (-P_inv_inc / diag_inc[np.newaxis, :]).astype(np.float32)
    np.fill_diagonal(B_inc_ease, 0)
    ease_inc_scores = {}
    for uid in target_uids:
        if uid in inc_ease_uidx:
            x_row = X_inc_ease[inc_ease_uidx[uid]].toarray().ravel()
            ease_inc_scores[uid] = x_row @ B_inc_ease
    del XtX_inc, P_inv_inc, B_inc_ease; gc.collect()

    log(f'[{now()}] prepare done in {(time.time() - t0):.0f}s')
    return {
        'pos': pos, 'before_pos': before_pos, 'incident_pos': incident_pos, 'after_pos': after_pos,
        'seen': seen, 'user_inc': user_inc, 'user_bef': user_bef, 'user_aft': user_aft,
        'b1': b1, 'mat': mat, 'uid2idx': uid2idx, 'iid2idx': iid2idx, 'idx2iid': idx2iid, 'idx2uid': idx2uid,
        'neighbors': neighbors, 'svd_scores': svd_scores, 'itf': itf,
        'ease_user_scores': ease_user_scores, 'ease_user_scores2': ease_user_scores2,
        'ease_item_idx': ease_item_idx, 'ease_item_rev': ease_item_rev,
        'ease_inc_scores': ease_inc_scores, 'inc_ease_idx': inc_ease_idx
    }

 # FEATURE GENERATION
 
def generate_feats_fast(inter_df, target_uids, feats, editions, users, inc_start, inc_end, prep, max_cands=700):
    t0 = time.time()
    item_author = feats['item_author']
    item_book = feats['item_book']
    ed_lang = feats['ed_lang']
    author_editions = feats['author_editions']
    item_genres = feats['item_genres']
    pos = prep['pos']; before_pos = prep['before_pos']; incident_pos = prep['incident_pos']; after_pos = prep['after_pos']
    seen = prep['seen']; user_inc = prep['user_inc']; user_bef = prep['user_bef']; user_aft = prep['user_aft']
    b1 = prep['b1']; mat = prep['mat']
    uid2idx = prep['uid2idx']; iid2idx = prep['iid2idx']; idx2iid = prep['idx2iid']; idx2uid = prep['idx2uid']
    neighbors = prep['neighbors']; svd_scores = prep['svd_scores']; itf = prep['itf']
    ease_user_scores = prep['ease_user_scores']; ease_user_scores2 = prep['ease_user_scores2']
    ease_item_idx = prep['ease_item_idx']; ease_item_rev = prep['ease_item_rev']
    ease_inc_scores = prep['ease_inc_scores']; inc_ease_idx = prep['inc_ease_idx']
    target_set = set(target_uids)

    log(f'[{now()}] Before: {len(before_pos):,} | Inc: {len(incident_pos):,} | After: {len(after_pos):,}')

    pop_inc_u = incident_pos.groupby('edition_id')['user_id'].nunique().to_dict()
    pop_bef_u = before_pos.groupby('edition_id')['user_id'].nunique().to_dict()
    pop_aft_u = after_pos.groupby('edition_id')['user_id'].nunique().to_dict()
    pop_inc_c = incident_pos['edition_id'].value_counts().to_dict()

    auth_df = pos[['user_id', 'edition_id', 'event_type']].copy()
    auth_df['author_id'] = auth_df['edition_id'].map(item_author)
    auth_df = auth_df[auth_df['author_id'].notna() & (auth_df['author_id'] != -1)].copy()
    auth_df['w'] = np.where(auth_df['event_type'].values == 2, 3.0, 1.0)
    user_author = auth_df.groupby(['user_id', 'author_id'])['w'].sum().reset_index()
    user_author_dict = {}
    for uid, g in user_author.groupby('user_id'):
        user_author_dict[uid] = dict(zip(g['author_id'].astype(int), g['w']))

    user_genres = {}
    for uid in target_uids:
        gc_ = Counter()
        for eid in seen.get(uid, set()):
            for g in item_genres.get(eid, set()):
                gc_[g] += 1
        total = sum(gc_.values())
        if total > 0:
            user_genres[uid] = {g: c / total for g, c in gc_.items()}

    user_lang = {}
    for uid in target_uids:
        langs = Counter()
        for e in seen.get(uid, set()):
            l = ed_lang.get(e)
            if l is not None:
                langs[l] += 1
        if langs:
            tl, tc = langs.most_common(1)[0]
            if tc / max(sum(langs.values()), 1) >= 0.7:
                user_lang[uid] = tl

    log(f'[{now()}] U2U scoring...')
    u2u_all = {}
    for uid in target_uids:
        if uid not in uid2idx:
            continue
        tidx = uid2idx[uid]
        if tidx not in neighbors:
            continue
        ni, ns = neighbors[tidx]
        seen_u = seen.get(uid, set())
        cands = defaultdict(lambda: [0.0, 0, 0.0])
        for n_i, n_s in zip(ni, ns):
            row = mat.getrow(n_i)
            for ii, iw in zip(row.indices, row.data):
                eid = idx2iid[ii]
                if eid in seen_u:
                    continue
                cd = cands[eid]
                cd[0] += float(n_s) * float(iw)
                cd[1] += 1
                cd[2] = max(cd[2], float(n_s))
        u2u_all[uid] = dict(sorted(cands.items(), key=lambda x: -x[1][0])[:1500])

    log(f'[{now()}] Surge...')
    pers_surge = {}
    for tidx, (ni, ns) in neighbors.items():
        uid = idx2uid.get(tidx)
        if uid is None or uid not in target_set:
            continue
        sc = defaultdict(float)
        for n_i, n_s in zip(ni, ns):
            nu = idx2uid.get(n_i)
            if nu is None:
                continue
            missing = (user_bef.get(nu, set()) | user_aft.get(nu, set())) - user_inc.get(nu, set())
            for eid in missing:
                sc[eid] += float(n_s)
        pers_surge[uid] = dict(sc)

    log(f'[{now()}] Bridge...')
    bridge = {}
    for tidx, (ni, ns) in neighbors.items():
        uid = idx2uid.get(tidx)
        if uid is None or uid not in target_set:
            continue
        seen_u = seen.get(uid, set())
        bef = user_bef.get(uid, set())
        if not bef:
            bridge[uid] = {}
            continue
        sc = defaultdict(float)
        for n_i, n_s in zip(ni, ns):
            nu = idx2uid.get(n_i)
            if nu is None:
                continue
            common = len(bef & user_bef.get(nu, set()))
            if common == 0:
                continue
            for eid in user_aft.get(nu, set()):
                if eid not in seen_u:
                    sc[eid] += float(n_s) * np.log1p(common)
        bridge[uid] = dict(sc)

    log(f'[{now()}] I2I...')
    inc_i2i = defaultdict(Counter)
    uii = incident_pos.groupby('user_id')['edition_id'].agg(set).to_dict()
    for uid_i, items in uii.items():
        if len(items) > 200:
            continue
        il = list(items)
        for i, a in enumerate(il):
            for b_item in il[i + 1:]:
                inc_i2i[a][b_item] += 1
                inc_i2i[b_item][a] += 1

    i2i_all = {}
    for uid in target_uids:
        seeds = user_inc.get(uid, set())
        seen_u = seen.get(uid, set())
        sc = Counter()
        for se in seeds:
            for co, cnt in inc_i2i.get(se, {}).items():
                if co not in seen_u:
                    sc[co] += cnt
        i2i_all[uid] = dict(sc)

    log(f'[{now()}] Continuation...')
    ed_users = defaultdict(set)
    for u, items in pos.groupby('user_id')['edition_id'].agg(set).to_dict().items():
        for e in items:
            ed_users[e].add(u)

    cont_all = {}
    for uid in target_uids:
        aft = user_aft.get(uid, set())
        if not aft:
            cont_all[uid] = {}
            continue
        sc = Counter()
        n = 0
        for eid in aft:
            for nu in list(ed_users.get(eid, set()))[:50]:
                if nu == uid:
                    continue
                for ce in user_inc.get(nu, set()):
                    sc[ce] += 1
                n += 1
        t = max(n, 1)
        cont_all[uid] = {e: c / t for e, c in sc.items()}

    reads = inter_df[(inter_df['event_type'] == 2) & (inter_df['rating'].notna())].copy()
    item_mean_rat = reads.groupby('edition_id')['rating'].mean().to_dict()
    item_rat_cnt = reads.groupby('edition_id')['rating'].size().to_dict()

    user_n = {uid: len(seen.get(uid, set())) for uid in target_uids}
    user_ni = {uid: len(user_inc.get(uid, set())) for uid in target_uids}
    user_nb = {uid: len(user_bef.get(uid, set())) for uid in target_uids}
    user_na = {uid: len(user_aft.get(uid, set())) for uid in target_uids}

    all_uid, all_eid = [], []
    for uid in target_uids:
        seen_u = seen.get(uid, set())
        cs = Counter()
        for eid, (sc, _, _) in u2u_all.get(uid, {}).items():
            cs[eid] += sc
        for eid, sc in i2i_all.get(uid, {}).items():
            cs[eid] += sc * 0.5
        for eid, sc in pers_surge.get(uid, {}).items():
            cs[eid] += sc * 0.3
        for eid, sc in cont_all.get(uid, {}).items():
            cs[eid] += sc * 100
        for eid, sc in bridge.get(uid, {}).items():
            cs[eid] += sc * 0.2
        ua = user_author_dict.get(uid, {})
        for aid, w in sorted(ua.items(), key=lambda x: -x[1])[:10]:
            for eid in author_editions.get(aid, set()):
                if eid not in seen_u:
                    cs[eid] += w * 0.1
        ease_sc = ease_user_scores.get(uid)
        if ease_sc is not None:
            top_ease = np.argsort(-ease_sc)[:300]
            for ei in top_ease:
                eid = ease_item_rev[ei]
                if eid not in seen_u and ease_sc[ei] > 0:
                    cs[eid] = max(cs.get(eid, 0), float(ease_sc[ei]) * 0.5)
        for eid, sc in b1['enr_ease'].get(uid, {}).items():
            if eid not in seen_u:
                cs[eid] = max(cs.get(eid, 0), sc * 0.45)
        for eid, sc in b1['enr_rrf'].get(uid, {}).items():
            if eid not in seen_u:
                cs[eid] += sc * 30.0
        top = [eid for eid, _ in cs.most_common(max_cands)]
        all_uid.extend([uid] * len(top))
        all_eid.extend(top)

    n_pairs = len(all_uid)
    log(f'[{now()}] Pairs: {n_pairs:,}')
    uid_arr = np.array(all_uid)
    eid_arr = np.array(all_eid)
    feat = np.zeros((n_pairs, 47), dtype=np.float32)

    inc_users_list = sorted(incident_pos['user_id'].unique())
    inc_items_list = sorted(incident_pos['edition_id'].unique())
    inc_u2i = {u: i for i, u in enumerate(inc_users_list)}
    inc_i2i_idx = {e: i for i, e in enumerate(inc_items_list)}

    svd_inc_scores = {}
    itf2 = None
    if len(incident_pos) > 500:
        ir = incident_pos['user_id'].map(inc_u2i).values
        ic = incident_pos['edition_id'].map(inc_i2i_idx).values
        iw = np.where(incident_pos['event_type'].values == 2, 3.0, 1.0).astype(np.float32)
        inc_mat = csr_matrix((iw, (ir, ic)), shape=(len(inc_users_list), len(inc_items_list)))
        inc_mat.data = np.log1p(inc_mat.data)
        k2 = min(24, min(inc_mat.shape) - 2)
        U2, s2, V2 = svds(inc_mat.astype(np.float64), k=k2)
        sq2 = np.sqrt(s2)
        uf2 = (U2 * sq2).astype(np.float32)
        itf2 = (V2.T * sq2).astype(np.float32)
        for uid in target_uids:
            if uid in inc_u2i:
                svd_inc_scores[uid] = uf2[inc_u2i[uid]]
        del inc_mat, U2, s2, V2; gc.collect()

    item_last_ts = inter_df.groupby('edition_id')['event_ts'].max().to_dict()
    ref_ts = inc_end
    n_b = max((before_pos['event_ts'].max() - before_pos['event_ts'].min()).days, 1) if len(before_pos) > 0 else 1
    n_i = max((incident_pos['event_ts'].max() - incident_pos['event_ts'].min()).days, 1) if len(incident_pos) > 0 else 1
    n_a = max((after_pos['event_ts'].max() - after_pos['event_ts'].min()).days, 1) if len(after_pos) > 0 else 1
    pop_bef_c = before_pos['edition_id'].value_counts().to_dict()
    pop_aft_c = after_pos['edition_id'].value_counts().to_dict()

    user_mx = {}
    for uid in target_uids:
        u2u = u2u_all.get(uid, {})
        mu = max((v[0] for v in u2u.values()), default=1.0)
        mi = max(i2i_all.get(uid, {}).values(), default=1.0)
        mp = max(pers_surge.get(uid, {}).values(), default=1.0)
        mc = max(cont_all.get(uid, {}).values(), default=1.0)
        mb = max(bridge.get(uid, {}).values(), default=1.0)
        user_mx[uid] = (mu, mi, mp, mc, mb)

    last_uid = None
    ease_row = None
    for idx in range(n_pairs):
        uid = uid_arr[idx]
        eid = eid_arr[idx]
        mu, mi, mp, mc, mb = user_mx[uid]
        if uid != last_uid:
            ease_row = ease_user_scores.get(uid)
            last_uid = uid
        u2u = u2u_all.get(uid, {}).get(eid)
        if u2u:
            feat[idx, 0] = u2u[0] / (mu + 1e-9)
            feat[idx, 1] = u2u[1]
            feat[idx, 2] = u2u[2]
            feat[idx, 3] = u2u[0]
        ii = i2i_all.get(uid, {}).get(eid, 0)
        feat[idx, 4] = ii / (mi + 1e-9)
        feat[idx, 5] = ii
        ps = pers_surge.get(uid, {}).get(eid, 0)
        feat[idx, 6] = ps / (mp + 1e-9)
        feat[idx, 7] = ps
        co = cont_all.get(uid, {}).get(eid, 0)
        feat[idx, 8] = co / (mc + 1e-9)
        br = bridge.get(uid, {}).get(eid, 0)
        feat[idx, 9] = br / (mb + 1e-9)
        feat[idx, 10] = pop_inc_u.get(eid, 0)
        feat[idx, 11] = pop_bef_u.get(eid, 0)
        feat[idx, 12] = pop_aft_u.get(eid, 0)
        feat[idx, 13] = pop_inc_c.get(eid, 0)
        rb = pop_bef_c.get(eid, 0) / n_b
        ri = pop_inc_c.get(eid, 0) / n_i
        ra = pop_aft_c.get(eid, 0) / n_a
        exp = (rb + ra) / 2
        feat[idx, 14] = max((exp - ri) / exp, 0) if exp > 0 else 0
        aid = item_author.get(eid)
        ua = user_author_dict.get(uid, {})
        if aid is not None and aid in ua:
            feat[idx, 15] = ua[aid]
            feat[idx, 16] = 1
        ug = user_genres.get(uid, {})
        ig = item_genres.get(eid, set())
        if ug and ig:
            feat[idx, 17] = sum(ug.get(g, 0) for g in ig)
        ul = user_lang.get(uid)
        el = ed_lang.get(eid)
        if ul is not None and el is not None:
            feat[idx, 18] = 1 if el == ul else 0
            feat[idx, 19] = 1 if el != ul else 0
        feat[idx, 20] = user_n.get(uid, 0)
        feat[idx, 21] = user_ni.get(uid, 0)
        feat[idx, 22] = user_nb.get(uid, 0)
        feat[idx, 23] = user_na.get(uid, 0)
        if uid in svd_scores and eid in iid2idx:
            feat[idx, 30] = float(np.dot(svd_scores[uid], itf[iid2idx[eid]]))
        feat[idx, 31] = item_mean_rat.get(eid, 0)
        feat[idx, 32] = item_rat_cnt.get(eid, 0)
        feat[idx, 33] = len(user_author_dict.get(uid, {}))
        n_src = 0
        if u2u_all.get(uid, {}).get(eid): n_src += 1
        if i2i_all.get(uid, {}).get(eid, 0) > 0: n_src += 1
        if pers_surge.get(uid, {}).get(eid, 0) > 0: n_src += 1
        if cont_all.get(uid, {}).get(eid, 0) > 0: n_src += 1
        if bridge.get(uid, {}).get(eid, 0) > 0: n_src += 1
        feat[idx, 34] = n_src
        if uid in svd_inc_scores and itf2 is not None and eid in inc_i2i_idx:
            feat[idx, 35] = float(np.dot(svd_inc_scores[uid], itf2[inc_i2i_idx[eid]]))
        feat[idx, 36] = feat[idx, 0] * feat[idx, 15]
        feat[idx, 37] = feat[idx, 30] * feat[idx, 4]
        feat[idx, 38] = feat[idx, 0] * feat[idx, 4]
        lts = item_last_ts.get(eid)
        feat[idx, 39] = max((ref_ts - lts).total_seconds() / 86400.0, 0) if lts is not None else 999
        if ease_row is not None and eid in ease_item_idx:
            feat[idx, 40] = ease_row[ease_item_idx[eid]]
        feat[idx, 41] = feat[idx, 40] * feat[idx, 15]
        feat[idx, 42] = feat[idx, 40] * feat[idx, 0]
        feat[idx, 43] = feat[idx, 40] * feat[idx, 17]
        ease_row2 = ease_user_scores2.get(uid)
        if ease_row2 is not None and eid in ease_item_idx:
            feat[idx, 44] = ease_row2[ease_item_idx[eid]]
        ease_inc = ease_inc_scores.get(uid)
        if ease_inc is not None and eid in inc_ease_idx:
            feat[idx, 45] = ease_inc[inc_ease_idx[eid]]
        feat[idx, 46] = feat[idx, 40] * feat[idx, 45]

    cols = [
        'u2u_score', 'u2u_n', 'u2u_maxsim', 'u2u_raw',
        'i2i_score', 'i2i_raw', 'surge_n', 'surge_raw', 'cont', 'bridge',
        'pop_inc_u', 'pop_bef_u', 'pop_aft_u', 'pop_inc_c', 'item_surge',
        'author_aff', 'known_author', 'genre_match',
        'lang_match', 'lang_mismatch',
        'user_n', 'user_ni', 'user_nb', 'user_na',
        'u2u_rank', 'i2i_rank', 'surge_rank', 'cont_rank', 'bridge_rank', 'heuristic',
        'svd_score', 'item_mean_rat', 'item_rat_cnt', 'user_n_authors', 'n_sources',
        'svd_inc_score', 'u2u_x_author', 'svd_x_i2i', 'u2u_x_i2i', 'item_recency',
        'ease_score', 'ease_x_author', 'ease_x_u2u', 'ease_x_genre',
        'ease2_score', 'ease_inc_score', 'ease_agreement'
    ]
    df = pd.DataFrame(feat, columns=cols)
    df.insert(0, 'edition_id', eid_arr)
    df.insert(0, 'user_id', uid_arr)
    for col in ['u2u_score', 'i2i_score', 'surge_n', 'cont', 'bridge']:
        df[f'{col}_rank'] = df.groupby('user_id')[col].rank(ascending=False, method='min')
    df['heuristic'] = (
        0.70 * df['u2u_score'] + 0.30 * df['i2i_score'] +
        0.30 * df['known_author'] * df['author_aff'] / 15.0 +
        0.12 * df['genre_match'] * 3.0 + 0.18 * df['surge_n'] +
        0.15 * df['cont'] + 0.15 * df['bridge']
    )
    pairs = list(zip(df['user_id'].values, df['edition_id'].values))
    df['enr_ease_score'] = np.array([b1['enr_ease'].get(u, {}).get(e, 0.0) for u, e in pairs], dtype=np.float32)
    df['enr_knn_score'] = np.array([b1['enr_knn'].get(u, {}).get(e, 0.0) for u, e in pairs], dtype=np.float32)
    df['enr_auth_score'] = np.array([b1['enr_auth'].get(u, {}).get(e, 0.0) for u, e in pairs], dtype=np.float32)
    df['enr_rrf_score'] = np.array([b1['enr_rrf'].get(u, {}).get(e, 0.0) for u, e in pairs], dtype=np.float32)
    df['from_enr_ease'] = (df['enr_ease_score'] > 0).astype(np.int8)
    df['from_enr_knn'] = (df['enr_knn_score'] > 0).astype(np.int8)
    df['from_enr_auth'] = (df['enr_auth_score'] > 0).astype(np.int8)
    df['from_enr_rrf'] = (df['enr_rrf_score'] > 0).astype(np.int8)
    df['enr_source_cnt'] = df[['from_enr_ease', 'from_enr_knn', 'from_enr_auth', 'from_enr_rrf']].sum(axis=1)
    for col in ['enr_ease_score', 'enr_knn_score', 'enr_auth_score', 'enr_rrf_score']:
        df[f'{col}_rank'] = df.groupby('user_id')[col].rank(ascending=False, method='min')
    df['heuristic_combined'] = (
        df['heuristic'] + 0.10 * df['enr_ease_score'] + 0.10 * df['enr_knn_score'] +
        0.08 * df['enr_auth_score'] + 0.20 * df['enr_rrf_score']
    )
    log(f'[{now()}] tabular features...')
    df = add_tabular_feats(df, inter_df, editions, users)
    log(f'[{now()}] done: {df.shape} in {(time.time() - t0):.0f}s')
    return df, seen, feats['item_book'], feats['item_author']

 # TABULAR FEATURES
 
def add_tabular_feats(df, past_inter, editions, users):
    meta_cols = ['edition_id', 'book_id', 'author_id', 'publisher_id', 'publication_year', 'age_restriction', 'language_id', 'genres_cnt']
    df = df.merge(editions[meta_cols], on='edition_id', how='left')
    user_cols = ['user_id']
    for col in ['gender', 'age_cat']:
        if col in users.columns:
            user_cols.append(col)
    df = df.merge(users[user_cols].drop_duplicates('user_id'), on='user_id', how='left')
    last_ts = past_inter['event_ts'].max()
    user_stats = past_inter.groupby('user_id').agg(
        user_cnt=('edition_id', 'size'),
        user_read_cnt=('event_type', lambda x: int((x == 2).sum())),
        user_wish_cnt=('event_type', lambda x: int((x == 1).sum())),
        user_last_inter=('event_ts', 'max')
    ).reset_index()
    user_stats['user_read_share'] = user_stats['user_read_cnt'] / np.maximum(user_stats['user_cnt'], 1)
    user_stats['user_days_since_last'] = (last_ts - user_stats['user_last_inter']).dt.total_seconds() / 86400.0
    user_stats = user_stats.drop(columns=['user_last_inter'])
    df = df.merge(user_stats, on='user_id', how='left')
    past = past_inter.copy()
    past['delta_days'] = (last_ts - past['event_ts']).dt.total_seconds() / 86400.0
    past['w7'] = np.exp(-past['delta_days'] / 7.0)
    past['w30'] = np.exp(-past['delta_days'] / 30.0)
    item_stats = past.groupby('edition_id').agg(
        item_popularity=('event_type', 'size'),
        item_read_cnt=('event_type', lambda x: int((x == 2).sum())),
        item_wish_cnt=('event_type', lambda x: int((x == 1).sum())),
        item_last_inter=('event_ts', 'max'),
        item_pop_decay_7=('w7', 'sum'),
        item_pop_decay_30=('w30', 'sum')
    ).reset_index()
    rated = past[past['rating'].fillna(0) > 0].copy()
    global_mean = float(rated['rating'].mean()) if len(rated) else 0.0
    if len(rated):
        br = rated.groupby('edition_id').agg(v=('rating', 'count'), R=('rating', 'mean')).reset_index()
        C = max(1.0, br['v'].quantile(0.25))
        br['item_bayes_rating'] = ((br['v'] * br['R']) + (C * global_mean)) / (br['v'] + C)
        item_stats = item_stats.merge(br[['edition_id', 'item_bayes_rating']], on='edition_id', how='left')
    else:
        item_stats['item_bayes_rating'] = global_mean
    item_stats['item_bayes_rating'] = item_stats['item_bayes_rating'].fillna(global_mean)
    item_stats['item_days_since_last'] = (last_ts - item_stats['item_last_inter']).dt.total_seconds() / 86400.0
    item_stats['item_popularity_log'] = np.log1p(item_stats['item_popularity'])
    item_stats = item_stats.drop(columns=['item_last_inter'])
    df = df.merge(item_stats, on='edition_id', how='left')
    past_aug = past_inter.merge(editions[['edition_id', 'book_id', 'author_id', 'publisher_id', 'language_id']], on='edition_id', how='left')
    past_aug['delta_days'] = (last_ts - past_aug['event_ts']).dt.total_seconds() / 86400.0
    past_aug['w30'] = np.exp(-past_aug['delta_days'] / 30.0)
    past30 = past_aug[past_aug['event_ts'] >= (last_ts - pd.Timedelta(days=30))].copy()
    item30 = past30.groupby('edition_id').size().rename('item_popularity_30').reset_index()
    book30 = past30.groupby('book_id').size().rename('book_popularity_30').reset_index()
    auth30 = past30.groupby('author_id').size().rename('author_popularity_30').reset_index()
    pub30 = past30.groupby('publisher_id').size().rename('publisher_popularity_30').reset_index()
    itemd30 = past_aug.groupby('edition_id')['w30'].sum().rename('item_pop_decay_30_all').reset_index()
    bookd30 = past_aug.groupby('book_id')['w30'].sum().rename('book_pop_decay_30').reset_index()
    authd30 = past_aug.groupby('author_id')['w30'].sum().rename('author_pop_decay_30').reset_index()
    pubd30 = past_aug.groupby('publisher_id')['w30'].sum().rename('publisher_pop_decay_30').reset_index()
    df = df.merge(item30, on='edition_id', how='left')
    df = df.merge(book30, on='book_id', how='left')
    df = df.merge(auth30, on='author_id', how='left')
    df = df.merge(pub30, on='publisher_id', how='left')
    df = df.merge(itemd30, on='edition_id', how='left')
    df = df.merge(bookd30, on='book_id', how='left')
    df = df.merge(authd30, on='author_id', how='left')
    df = df.merge(pubd30, on='publisher_id', how='left')
    for ent, pref in [('author_id', 'ua'), ('book_id', 'ub'), ('publisher_id', 'up')]:
        tmp = past_aug.groupby(['user_id', ent]).agg(
            cnt=('edition_id', 'size'),
            read_cnt=('event_type', lambda x: int((x == 2).sum())),
            wish_cnt=('event_type', lambda x: int((x == 1).sum())),
            last_ts=('event_ts', 'max')
        ).reset_index()
        tmp[f'{pref}_days_since_last'] = (last_ts - tmp['last_ts']).dt.total_seconds() / 86400.0
        tmp = tmp.drop(columns=['last_ts'])
        tmp = tmp.rename(columns={'cnt': f'{pref}_cnt', 'read_cnt': f'{pref}_read_cnt', 'wish_cnt': f'{pref}_wish_cnt'})
        df = df.merge(tmp, on=['user_id', ent], how='left')
    lang_pref = past_aug.groupby(['user_id', 'language_id']).size().rename('cnt').reset_index()
    if len(lang_pref):
        idx = lang_pref.groupby('user_id')['cnt'].idxmax()
        top_lang = lang_pref.loc[idx, ['user_id', 'language_id']].rename(columns={'language_id': 'user_top_lang'})
        df = df.merge(top_lang, on='user_id', how='left')
        df['lang_eq_top'] = (df['language_id'] == df['user_top_lang']).astype(np.int8)
    else:
        df['user_top_lang'] = -1
        df['lang_eq_top'] = 0
    log_cols = ['item_popularity_30', 'book_popularity_30', 'author_popularity_30', 'publisher_popularity_30']
    for c in log_cols:
        df[c] = df[c].fillna(0)
        df[f'{c}_log'] = np.log1p(df[c])
    num_fill_zero = [
        'user_cnt', 'user_read_cnt', 'user_wish_cnt', 'user_read_share', 'user_days_since_last',
        'item_popularity', 'item_read_cnt', 'item_wish_cnt', 'item_pop_decay_7', 'item_pop_decay_30',
        'item_bayes_rating', 'item_days_since_last', 'item_popularity_log',
        'item_popularity_30', 'book_popularity_30', 'author_popularity_30', 'publisher_popularity_30',
        'item_pop_decay_30_all', 'book_pop_decay_30', 'author_pop_decay_30', 'publisher_pop_decay_30',
        'ua_cnt', 'ua_read_cnt', 'ua_wish_cnt', 'ua_days_since_last',
        'ub_cnt', 'ub_read_cnt', 'ub_wish_cnt', 'ub_days_since_last',
        'up_cnt', 'up_read_cnt', 'up_wish_cnt', 'up_days_since_last', 'lang_eq_top'
    ]
    for c in num_fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    cat_fill = {'gender': -1, 'age_cat': 'unk', 'user_top_lang': -1, 'publisher_id': -1,
        'publication_year': 0, 'age_restriction': -1, 'language_id': -1, 'book_id': -1, 'author_id': -1}
    for c, v in cat_fill.items():
        if c in df.columns:
            df[c] = df[c].fillna(v)
    return df

 # RANKER
 
def train_lgb_ranker(train_df, val_df):
    cat_cols = ['user_id', 'edition_id', 'book_id', 'author_id', 'publisher_id',
        'language_id', 'age_restriction', 'gender', 'age_cat', 'user_top_lang']
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    feat_cols = [c for c in train_df.columns if c not in ['label', 'query_id']]
    train = train_df.copy()
    val = val_df.copy()
    for df in [train, val]:
        for c in feat_cols:
            if c in cat_cols:
                df[c] = df[c].fillna('unk').astype('category')
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype(np.float32)
    train_group = train.groupby('user_id', sort=False).size().to_numpy()
    val_group = val.groupby('user_id', sort=False).size().to_numpy()
    train_data = lgb.Dataset(train[feat_cols], label=train['label'].astype(np.float32),
        group=train_group, categorical_feature=cat_cols, free_raw_data=False)
    val_data = lgb.Dataset(val[feat_cols], label=val['label'].astype(np.float32),
        group=val_group, categorical_feature=cat_cols, reference=train_data, free_raw_data=False)
    models = []
    for seed in [42]:
        params = {
            'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [20],
            'learning_rate': 0.03, 'num_leaves': 127, 'min_data_in_leaf': 30,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
            'lambda_l2': 10.0, 'verbosity': -1, 'seed': seed, 'force_row_wise': True
        }
        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(250)]
        model = lgb.train(params, train_data, num_boost_round=6000,
            valid_sets=[val_data], callbacks=callbacks)
        models.append(model)
    return models, feat_cols, cat_cols

def predict_lgb(models, df, feat_cols, cat_cols):
    data = df.copy()
    for c in feat_cols:
        if c in cat_cols:
            data[c] = data[c].fillna('unk').astype('category')
        else:
            data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0.0).astype(np.float32)
    preds = np.zeros(len(data), dtype=np.float32)
    for m in models:
        preds += m.predict(data[feat_cols], num_iteration=m.best_iteration).astype(np.float32)
    preds /= len(models)
    return preds

def rank_top20(df, scores, target_uids, seen, item_book, item_author, fallback_items):
    data = df.copy()
    data['score'] = scores
    recs = []
    for uid in target_uids:
        ud = data[data['user_id'] == uid].nlargest(100, 'score')
        seen_u = seen.get(uid, set())
        sel = []
        used_books = set()
        author_cnt = defaultdict(int)
        for _, r in ud.iterrows():
            if len(sel) >= 20:
                break
            eid = r['edition_id']
            bid = item_book.get(eid)
            aid = item_author.get(eid)
            if bid is not None and bid in used_books:
                continue
            if aid is not None and aid != -1 and author_cnt[aid] >= 3:
                continue
            sel.append(eid)
            if bid is not None:
                used_books.add(bid)
            if aid is not None and aid != -1:
                author_cnt[aid] += 1
        if len(sel) < 20:
            for eid in fallback_items:
                if eid not in seen_u and eid not in set(sel):
                    sel.append(eid)
                    if len(sel) >= 20:
                        break
        for rank, eid in enumerate(sel[:20], 1):
            recs.append((uid, eid, rank))
    return pd.DataFrame(recs, columns=['user_id', 'edition_id', 'rank'])

 # MAIN
 
if __name__ == '__main__':
    T0 = time.time()

    log(f'[{now()}] Loading data...')
    interactions = pd.read_csv(os.path.join(DATA_DIR, 'interactions.csv'))
    interactions['event_ts'] = pd.to_datetime(interactions['event_ts'])
    targets = pd.read_csv(os.path.join(DATA_DIR, 'targets.csv'))
    editions = pd.read_csv(os.path.join(DATA_DIR, 'editions.csv'))
    users = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    book_genres = pd.read_csv(os.path.join(DATA_DIR, 'book_genres.csv'))

    hack_path = os.path.join(DATA_DIR, 'hack_interactions.csv')
    if not os.path.exists(hack_path):
        hack_path = os.path.join(os.environ.get('HACK_PATH', 'data-old-hackathon'), 'interactions.csv')
    hack_interactions = pd.read_csv(hack_path)
    hack_interactions['event_ts'] = pd.to_datetime(hack_interactions['event_ts'])

    target_uids = targets['user_id'].tolist()

    editions = editions.drop(columns=['title', 'description'], errors='ignore')
    book2genres = book_genres.groupby('book_id')['genre_id'].apply(list).reset_index()
    editions = editions.merge(book2genres, on='book_id', how='left')
    editions['genre_id'] = editions['genre_id'].apply(lambda x: x if isinstance(x, list) else [])
    editions['genres_cnt'] = editions['genre_id'].apply(len).astype(int)
    for col in ['publisher_id', 'publication_year', 'age_restriction', 'language_id', 'author_id', 'book_id']:
        fill = 0 if col == 'publication_year' else -1
        editions[col] = editions[col].fillna(fill).astype(int)
    users['gender'] = users['gender'].fillna(-1).astype(int)
    age = users['age'].fillna(-1)
    users['age_cat'] = pd.cut(age, bins=[-2, 0, 18, 25, 35, 45, 60, 200],
        labels=['unk', '0_18', '18_25', '25_35', '35_45', '45_60', '60+']).astype(str)

    log(f'[{now()}] Data: {interactions.shape}, targets: {len(target_uids)}, hack: {hack_interactions.shape}')

    feats = precompute_feats(editions, book_genres)

    # ─── GENERATE FOLDS ───
    fold_dfs = []
    for fi, (cutoff, inc_s, inc_e) in enumerate(FOLDS):
        log(f'\n[{now()}] FOLD {fi+1}: cutoff={cutoff.date()}, incident=[{inc_s.date()}, {inc_e.date()})')
        inter_fold = interactions[interactions['event_ts'] < cutoff].copy()
        hack_fold = slice_by_end(hack_interactions, cutoff)
        vis, hs, hbu, fold_tgt = simulate_hidden(inter_fold, inc_s, inc_e, seed=SEED)
        if not fold_tgt:
            log(f'  Skip: no hidden items')
            continue
        prep = prepare_sources(vis, hack_fold, fold_tgt, feats, editions, inc_s, inc_e)
        df_feat, _, _, _ = generate_feats_fast(vis, fold_tgt, feats, editions, users, inc_s, inc_e, prep, max_cands=MAX_CANDS)
        labels = []
        for u, e in zip(df_feat['user_id'], df_feat['edition_id']):
            labels.append(1 if e in hbu.get(u, set()) else 0)
        df_feat['label'] = labels
        fold_dfs.append(df_feat)
        log(f'  Fold {fi+1}: {df_feat.shape}, positives: {sum(labels)}')
        del prep; gc.collect()

    # ─── TRAIN ───
    log(f'\n[{now()}] Training LightGBM...')
    val_df = fold_dfs[1]  # fold 2 = validation
    train_df = pd.concat([fold_dfs[0], fold_dfs[2]], ignore_index=True) if len(fold_dfs) > 2 else fold_dfs[0]
    train_df = train_df.sort_values('user_id').reset_index(drop=True)
    if 'query_id' in train_df.columns:
        train_df.drop(columns='query_id', inplace=True)
    if 'query_id' in val_df.columns:
        val_df.drop(columns='query_id', inplace=True)
    models, feat_cols, cat_cols = train_lgb_ranker(train_df, val_df)
    del train_df, val_df, fold_dfs; gc.collect()

    # ─── INFERENCE ───
    log(f'\n[{now()}] Generating test features...')
    prep_te = prepare_sources(interactions, hack_interactions, target_uids, feats, editions, INCIDENT_START, INCIDENT_END)
    te_feat, te_seen, te_ib, te_ia = generate_feats_fast(
        interactions, target_uids, feats, editions, users, INCIDENT_START, INCIDENT_END, prep_te, max_cands=MAX_CANDS)

    log(f'[{now()}] Predicting...')
    te_scores = predict_lgb(models, te_feat, feat_cols, cat_cols)
    fb_test = global_fallback(interactions, INCIDENT_START, INCIDENT_END)
    sub = rank_top20(te_feat, te_scores, target_uids, te_seen, te_ib, te_ia, fb_test)

    # pad missing
    pred_users = set(sub['user_id'])
    extra = []
    for uid in target_uids:
        cur = set(sub[sub['user_id'] == uid]['edition_id']) if uid in pred_users else set()
        if len(cur) >= 20:
            continue
        seen_u = te_seen.get(uid, set())
        rank = len(cur) + 1
        for eid in fb_test:
            if eid not in cur and eid not in seen_u:
                extra.append((uid, eid, rank))
                cur.add(eid)
                rank += 1
                if rank > 20:
                    break
    if extra:
        sub = pd.concat([sub, pd.DataFrame(extra, columns=['user_id', 'edition_id', 'rank'])], ignore_index=True)

    sub['edition_id'] = sub['edition_id'].astype(int)
    sub.to_csv('sub_lgb_v4.csv', index=False)
    log(f'\n[{now()}] Saved sub_lgb_v4.csv: {sub.shape}, users={sub["user_id"].nunique()}')
    log(f'Total time: {(time.time() - T0) / 60:.1f} min')

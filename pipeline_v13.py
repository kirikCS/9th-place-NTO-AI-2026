"""
v13 — Merged EASE+ALS+SVD -> CatBoost Ranker (binary labels)
============================================================
GPU-accelerated. Self-contained single script.

Architecture:
  1. Candidate generation (EASE@400+ALS@400+SVD@400+KNN+Author+Popular+Echo -> ~400/user)
  2. Feature engineering (~130 features)
  3. CatBoost YetiRank with 4 folds (main+echo) + fold decay (GPU)
  4. Binary labels (positive vs negative) — task = predict lost positives
"""
import os, re, gc, time, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix, hstack, eye as speye
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from implicit.als import AlternatingLeastSquares
from catboost import CatBoostRanker, Pool
import torch
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
USE_GPU = torch.cuda.is_available()
print(f'USE_GPU: {USE_GPU}')

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
DATA_PATH    = os.environ.get('DATA_DIR', 'data')
HACK_PATH    = os.environ.get('HACK_PATH', 'data-old-hackathon')
CACHE_DIR    = 'cache_v13'
INCIDENT_START = pd.Timestamp('2025-10-01')
INCIDENT_END   = pd.Timestamp('2025-11-01')

EASE_LAM     = 25
EASE_MIN_POP = 5
ALS_FACTORS  = 128
ALS_ITER     = 30
SVD_FACTORS  = 48
SVD2_FACTORS = 20

EMB_DIM      = 32
NEG_RATIOS   = [50, 80, 100, 120, 150, 180, 190, 200]
HARD_NEG_FRAC = 0.5

CB_DEPTH   = 8
CB_ITER    = 15000
CB_LR      = 0.03
CB_L2      = 14.0
CB_OD_WAIT = 350

os.makedirs(CACHE_DIR, exist_ok=True)

def save_cache(name, obj):
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = os.path.getsize(path) / 1024 / 1024
    print(f'  [CACHE] Saved {name} ({sz:.1f} MB)')

def load_cache(name):
    path = os.path.join(CACHE_DIR, f'{name}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        sz = os.path.getsize(path) / 1024 / 1024
        print(f'  [CACHE] Loaded {name} ({sz:.1f} MB)')
        return obj
    return None

def has_cache(name):
    return os.path.exists(os.path.join(CACHE_DIR, f'{name}.pkl'))

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print("=" * 60, "\nLOADING DATA\n", "=" * 60)
interactions = pd.read_csv(os.path.join(DATA_PATH, 'interactions.csv'))
interactions['event_ts'] = pd.to_datetime(interactions['event_ts'])
editions = pd.read_csv(os.path.join(DATA_PATH, 'editions.csv'))
users = pd.read_csv(os.path.join(DATA_PATH, 'users.csv'))
targets = pd.read_csv(os.path.join(DATA_PATH, 'targets.csv'))
book_genres = pd.read_csv(os.path.join(DATA_PATH, 'book_genres.csv'))
target_users = targets['user_id'].tolist()
target_set = set(target_users)

hack = None
if os.path.isdir(HACK_PATH):
    hack = pd.read_csv(os.path.join(HACK_PATH, 'interactions.csv'))
    hack['event_ts'] = pd.to_datetime(hack['event_ts'])
    print(f'Hack data: {len(hack)}')

# Genre cleanup
book_genres['genre_id'] = book_genres['genre_id'].replace({10: 468, 480: 479})
book_genres = book_genres[book_genres['genre_id'] != 139].drop_duplicates(['book_id', 'genre_id'])

# Text columns saved separately
editions_wtext = editions[['edition_id', 'title', 'description']].copy()
editions.drop(columns=['title', 'description'], inplace=True, errors='ignore')

# Genre list per edition
bg_grouped = book_genres.groupby('book_id')['genre_id'].apply(list).reset_index()
editions = editions.merge(bg_grouped, on='book_id', how='left')
editions['genres'] = editions['genre_id'].apply(lambda x: x if isinstance(x, list) else [])
editions['genres_cnt'] = editions['genres'].apply(len)
editions.drop(columns='genre_id', inplace=True, errors='ignore')

# Fill NaN
editions['publication_year'] = editions['publication_year'].fillna(0).astype(int)
editions['author_id'] = editions['author_id'].fillna(-1).astype(int)
editions['language_id'] = editions['language_id'].fillna(-1).astype(int)
editions['age_restriction'] = editions['age_restriction'].fillna(0).astype(int)
editions['publisher_id'] = editions['publisher_id'].fillna(-1).astype(int)

# User demographics
users['gender'] = users['gender'].fillna(0).astype(int)
users['age'] = users['age'].fillna(0).astype(float)
min_age = users.loc[users['age'] > 0, 'age'].min() if (users['age'] > 0).any() else 1
max_age = users['age'].max()
age_bins = [-1, min_age - 1, 12, 17, 24, 34, 44, 54, max_age + 1]
users['age_cat'] = pd.cut(users['age'], bins=age_bins, labels=False).fillna(0).astype(int)

# Lookup structures
ed_info = editions.set_index('edition_id')
lang_119 = set(editions[editions['language_id'] == 119]['edition_id'])
author_editions = editions[editions['author_id'] != -1].groupby('author_id')['edition_id'].apply(set).to_dict()
edition_book = editions.set_index('edition_id')['book_id'].to_dict()

print(f'Interactions: {len(interactions)}, Editions: {len(editions)}, Targets: {len(targets)}')

# ═══════════════════════════════════════════════════════════════
# TEXT EMBEDDINGS (SBERT + PCA)  [CACHED]
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60, "\nTEXT EMBEDDINGS\n", "=" * 60)

def _clean(text):
    if isinstance(text, str):
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    return ''

_emb_cache = load_cache('sbert_embeddings')
if _emb_cache is not None:
    embeddings = _emb_cache
    emb_cols = [c for c in embeddings.columns if c.startswith('emb_')]
    print(f'SBERT Embeddings (cached): {embeddings.shape}')
    del editions_wtext
else:
    device = 'cuda' if USE_GPU else 'cpu'
    st_model = SentenceTransformer('cointegrated/rubert-tiny2', device=device)
    texts_clean = (editions_wtext['title'].fillna('').map(_clean) + ' ' +
                   editions_wtext['description'].fillna('').map(_clean)).tolist()
    vecs = st_model.encode(texts_clean, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)

    pca = PCA(n_components=EMB_DIM, random_state=42)
    vecs_pca = pca.fit_transform(vecs)
    vecs_pca = vecs_pca / (np.linalg.norm(vecs_pca, axis=1, keepdims=True) + 1e-9)

    emb_cols = [f'emb_{i}' for i in range(EMB_DIM)]
    embeddings = pd.DataFrame(vecs_pca, columns=emb_cols)
    embeddings['edition_id'] = editions_wtext['edition_id'].values
    print(f'SBERT Embeddings: {embeddings.shape}')

    save_cache('sbert_embeddings', embeddings)

    del vecs, vecs_pca, st_model, editions_wtext
    gc.collect()

# ═══════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════

def train_ease(inter_df, hack_df=None, min_pop=EASE_MIN_POP, lam=EASE_LAM):
    pairs = inter_df[['user_id', 'edition_id']].drop_duplicates()
    if hack_df is not None:
        pairs = pd.concat([pairs, hack_df[['user_id', 'edition_id']].drop_duplicates()]).drop_duplicates()
    uids = sorted(pairs['user_id'].unique())
    iids = sorted(pairs['edition_id'].unique())
    u2i = {u: i for i, u in enumerate(uids)}
    i2i = {e: i for i, e in enumerate(iids)}
    rows = pairs['user_id'].map(u2i).values
    cols = pairs['edition_id'].map(i2i).values
    mat = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                     shape=(len(uids), len(iids)))
    pop = np.array(mat.sum(axis=0)).flatten()
    idx = np.where(pop >= min_pop)[0]
    eids = [iids[i] for i in idx]
    X = mat[:, idx].toarray().astype(np.float32)
    G = X.T @ X + lam * np.eye(len(idx), dtype=np.float32)
    B = np.linalg.inv(G)
    d = np.diag(B).copy(); B /= (-d[None, :]); np.fill_diagonal(B, 0)
    scores = X @ B
    del G, B, d, X; gc.collect()
    return scores, eids, {e: i for i, e in enumerate(eids)}, mat, uids, iids, u2i, i2i


def train_als(inter_df, factors=ALS_FACTORS, iters=ALS_ITER):
    adf = inter_df[['user_id', 'edition_id', 'event_type']].copy()
    adf['w'] = adf['event_type'].map({2: 5.0, 1: 2.0}).fillna(1.0)
    ag = adf.groupby(['user_id', 'edition_id'])['w'].max().reset_index()
    uc = ag['user_id'].astype('category'); ic = ag['edition_id'].astype('category')
    u2i = {u: i for i, u in enumerate(uc.cat.categories)}
    i2i = {e: i for i, e in enumerate(ic.cat.categories)}
    i2e = {i: e for e, i in i2i.items()}
    amat = csr_matrix((ag['w'].values.astype(np.float32),
                       (uc.cat.codes.values, ic.cat.codes.values)),
                      shape=(len(u2i), len(i2i)))
    als = AlternatingLeastSquares(factors=factors, regularization=0.01,
                                  iterations=iters, random_state=42, use_gpu=USE_GPU)
    als.fit(amat)
    uf = als.user_factors; itf = als.item_factors
    if hasattr(uf, 'to_numpy'): uf = uf.to_numpy(); itf = itf.to_numpy()
    return uf, itf, u2i, i2i, i2e, amat, als


def train_als_domain(inter_df, editions_df, domain_col, factors=64):
    meta = editions_df[['edition_id', domain_col]].dropna(subset=[domain_col])
    enriched = inter_df[['user_id', 'edition_id', 'event_type']].merge(meta, on='edition_id')
    enriched['w'] = enriched['event_type'].map({2: 5.0, 1: 2.0}).fillna(1.0)
    grouped = enriched.groupby(['user_id', domain_col])['w'].max().reset_index()
    uc = grouped['user_id'].astype('category'); ic = grouped[domain_col].astype('category')
    u2i = {u: i for i, u in enumerate(uc.cat.categories)}
    i2i = {e: i for i, e in enumerate(ic.cat.categories)}
    sp = csr_matrix((grouped['w'].values.astype(np.float32),
                     (uc.cat.codes.values, ic.cat.codes.values)),
                    shape=(len(u2i), len(i2i)))
    als = AlternatingLeastSquares(factors=factors, regularization=0.01,
                                  iterations=20, random_state=42, use_gpu=USE_GPU)
    als.fit(sp)
    uf = als.user_factors; itf = als.item_factors
    if hasattr(uf, 'to_numpy'): uf = uf.to_numpy(); itf = itf.to_numpy()
    return uf, itf, u2i, i2i


def train_svd_cf(inter_df, n_factors=SVD_FACTORS, use_log=True):
    inter = inter_df.copy()
    inter['w'] = inter['event_type'].map({1: 1.0, 2: 3.0}).fillna(0.0)
    agg = inter.groupby(['user_id', 'edition_id'])['w'].sum().reset_index()
    user_ids = sorted(agg['user_id'].unique()); item_ids = sorted(agg['edition_id'].unique())
    uid_map = {u: i for i, u in enumerate(user_ids)}
    iid_map = {e: i for i, e in enumerate(item_ids)}
    rows = agg['user_id'].map(uid_map).values
    cols = agg['edition_id'].map(iid_map).values
    vals = np.log1p(agg['w'].values) if use_log else agg['w'].values
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    k = min(n_factors, min(len(user_ids), len(item_ids)) - 1)
    if k <= 0: k = 1
    U, S, Vt = svds(matrix.astype(np.float64), k=k)
    idx = np.argsort(-S); U, S, Vt = U[:, idx], S[idx], Vt[idx, :]
    sqrt_S = np.sqrt(S)
    user_emb = U * sqrt_S[np.newaxis, :]
    item_emb = Vt.T * sqrt_S[np.newaxis, :]
    ue = {uid: user_emb[i] for uid, i in uid_map.items()}
    ie = {eid: item_emb[i] for eid, i in iid_map.items()}
    print(f'  SVD done: {len(user_ids)} users, {len(item_ids)} items, {k} factors')
    return ue, ie, k





def train_knn(mat, iids, min_pop=3):
    pop = np.array(mat.sum(axis=0)).flatten()
    pidx = np.where(pop >= min_pop)[0]
    peids = [iids[i] for i in pidx]
    vecs = mat[:, pidx].T.tocsr()
    nn = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(vecs)
    ds, ix = nn.kneighbors(vecs)
    knn = {}
    for i, eid in enumerate(peids):
        knn[eid] = [(peids[ix[i][j]], 1 - ds[i][j]) for j in range(50)]
    return knn


# ═══════════════════════════════════════════════════════════════
# CANDIDATE GENERATION (8 sources)
# ═══════════════════════════════════════════════════════════════


def generate_svd_candidates(svd_ue, svd_ie, target_uids, observed, n_recs=400):
    """SVD-based collaborative candidates."""
    print('  [SVD] Generating candidates...')
    all_eids = list(svd_ie.keys())
    item_mat = np.array([svd_ie[e] for e in all_eids])
    cands = {}
    for uid in target_uids:
        if uid not in svd_ue:
            cands[uid] = set()
            continue
        obs = observed.get(uid, set())
        scores = svd_ue[uid] @ item_mat.T
        top_idx = np.argsort(-scores)
        picked = set()
        for j in top_idx:
            eid = all_eids[j]
            if eid not in obs and eid not in picked:
                picked.add(eid)
            if len(picked) >= n_recs:
                break
        cands[uid] = picked
    print(f'    → {sum(len(v) for v in cands.values())} SVD candidates')
    return cands


def generate_all_candidates(target_uids, inter_df, hack_df, editions_df,
                             embeddings_df, book_genres_df, incident_start,
                             echo_start=None, echo_end=None):
    """Generate candidates from EASE@400+ALS@400+SVD@400+KNN+Author+Popular+Echo, cap at 400/user."""
    print('\n=== CANDIDATE GENERATION ===')
    t0 = time.time()

    # Observed interactions per user
    observed = defaultdict(set)
    for u, e in zip(inter_df['user_id'], inter_df['edition_id']):
        observed[u].add(e)

    # Author count per user
    ed_info_local = editions_df.set_index('edition_id')
    user_author_cnt = defaultdict(lambda: defaultdict(int))
    for u, e in zip(inter_df['user_id'], inter_df['edition_id']):
        if u in target_set and e in ed_info_local.index:
            a = ed_info_local.loc[e, 'author_id']
            if a != -1:
                user_author_cnt[u][a] += 1

    # 1. EASE
    print('  [1] EASE...')
    ease_data = train_ease(inter_df, hack_df)
    ease_sc, ease_eids, ease_e2i, ease_mat, ease_uids, ease_iids, ease_u2i, ease_i2i = ease_data

    # 2. ALS
    print('  [2] ALS...')
    als_data = train_als(inter_df)
    als_uf, als_if, als_u2i, als_i2i, als_i2e, als_mat, als_model = als_data

    # 3. SVD
    print('  [3] SVD...')
    svd_ue, svd_ie, svd_k = train_svd_cf(inter_df, n_factors=SVD_FACTORS, use_log=True)

    # 4. KNN
    print('  [4] ItemKNN...')
    knn_nb = train_knn(ease_mat, ease_iids)

    all_cands = {uid: set() for uid in target_uids}

    # Source 1: EASE top-200
    for uid in target_uids:
        if uid not in ease_u2i:
            continue
        uidx = ease_u2i[uid]
        user_items = {ease_iids[j] for j in ease_mat[uidx].indices}
        sc = ease_sc[uidx]
        ec = [(ease_eids[j], float(sc[j])) for j in range(len(ease_eids))
              if ease_eids[j] not in user_items]
        ec.sort(key=lambda x: -x[1])
        for e, _ in ec[:400]:
            all_cands[uid].add(e)

    # Source 2: ALS top-100
    for uid in target_uids:
        if uid not in als_u2i:
            continue
        uf = als_uf[als_u2i[uid]]
        dot = uf @ als_if.T
        obs = observed.get(uid, set())
        als_obs = {als_i2i[e] for e in obs if e in als_i2i}
        top_idx = np.argsort(-dot)
        cnt = 0
        for idx in top_idx:
            if idx in als_obs:
                continue
            eid = als_i2e.get(idx)
            if eid and eid not in obs:
                all_cands[uid].add(eid)
                cnt += 1
            if cnt >= 400:
                break

    # Source 3: SVD top-400
    svd_cands = generate_svd_candidates(svd_ue, svd_ie, target_uids, observed, n_recs=400)
    for uid in target_uids:
        all_cands[uid] |= svd_cands.get(uid, set())

    # Source 5: ItemKNN top-50
    for uid in target_uids:
        if uid not in ease_u2i:
            continue
        user_items = {ease_iids[j] for j in ease_mat[ease_u2i[uid]].indices}
        knn_scores = defaultdict(float)
        for e in user_items:
            if e in knn_nb:
                for ne, sim in knn_nb[e]:
                    knn_scores[ne] += sim
        obs = observed.get(uid, set())
        top_knn = sorted(knn_scores.items(), key=lambda x: -x[1])
        cnt = 0
        for eid, _ in top_knn:
            if eid not in obs:
                all_cands[uid].add(eid)
                cnt += 1
            if cnt >= 50:
                break

    # Source 6: Author continuation (≥2 books)
    author_eds = editions_df[editions_df['author_id'] != -1].groupby('author_id')['edition_id'].apply(set).to_dict()
    for uid in target_uids:
        obs = observed.get(uid, set())
        ua = user_author_cnt.get(uid, {})
        for aid, cnt in ua.items():
            if cnt >= 2 and aid in author_eds:
                for eid in author_eds[aid]:
                    if eid not in obs:
                        all_cands[uid].add(eid)

    # Source 7: Popular in incident window
    incident_inter = inter_df[inter_df['event_ts'] >= incident_start]
    pop_items = incident_inter.groupby('edition_id').size().nlargest(200).index.tolist()
    for uid in target_uids:
        obs = observed.get(uid, set())
        for eid in pop_items:
            if eid not in obs:
                all_cands[uid].add(eid)

    # Source 8: November Echo -- authors active in echo month — authors active in echo month
    if echo_start is not None and echo_end is not None:
        echo_inter = inter_df[(inter_df['event_ts'] >= echo_start) &
                               (inter_df['event_ts'] < echo_end)]
        echo_meta = echo_inter.merge(
            editions_df[['edition_id', 'author_id']], on='edition_id', how='left')
        echo_ua = echo_meta[echo_meta['user_id'].isin(set(target_uids))].groupby(
            ['user_id', 'author_id']).size().reset_index(name='echo_cnt')
        author_eds_local = editions_df[editions_df['author_id'] != -1].groupby(
            'author_id')['edition_id'].apply(set).to_dict()
        echo_added = 0
        for _, row in echo_ua.iterrows():
            uid, aid = int(row['user_id']), int(row['author_id'])
            if uid not in all_cands or aid not in author_eds_local:
                continue
            obs = observed.get(uid, set())
            for eid in author_eds_local[aid]:
                if eid not in obs:
                    all_cands[uid].add(eid)
                    echo_added += 1
        print(f'  [9] Echo candidates: {echo_added}')

    # Cap at 400/user — prioritize by EASE score (best proxy for relevance)
    for uid in target_uids:
        cands = list(all_cands[uid])
        if len(cands) > 400:
            if uid in ease_u2i:
                sc = ease_sc[ease_u2i[uid]]
                cands.sort(key=lambda e: -float(sc[ease_e2i[e]]) if e in ease_e2i else 0.0)
            all_cands[uid] = set(cands[:400])

    avg = np.mean([len(c) for c in all_cands.values()])
    print(f'\n  Total candidates: {sum(len(c) for c in all_cands.values())}')
    print(f'  Avg/user: {avg:.0f}, Time: {time.time()-t0:.0f}s')

    return all_cands, ease_data, als_data, svd_ue, svd_ie, svd_k, knn_nb


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (~155 features)
# ═══════════════════════════════════════════════════════════════

N_SVD_EW = 10  # element-wise product dims to keep

def build_features_df(cands_dict, inter_df, editions_df, users_df, book_genres_df,
                      embeddings_df, ease_data, als_data,
                      als_book, als_auth, als_pub,
                      svd_ue1, svd_ie1, svd_k1,
                      svd_ue2, svd_ie2, svd_k2,
                      knn_nb,
                      echo_start=None, echo_end=None):
    """Build feature DataFrame for all (user, candidate) pairs."""
    ease_sc, ease_eids, ease_e2i, ease_mat, ease_uids, ease_iids, ease_u2i, ease_i2i = ease_data
    als_uf, als_if, als_u2i, als_i2i, als_i2e, als_mat, als_model = als_data
    als_b_uf, als_b_if, als_b_u2i, als_b_i2i = als_book
    als_a_uf, als_a_if, als_a_u2i, als_a_i2i = als_auth
    als_p_uf, als_p_if, als_p_u2i, als_p_i2i = als_pub

    last_ts = inter_df['event_ts'].max()
    emb_c = [c for c in embeddings_df.columns if c.startswith('emb_')]

    # Flat pairs
    rows = [(uid, eid) for uid, cands in cands_dict.items() for eid in cands]
    df = pd.DataFrame(rows, columns=['user_id', 'edition_id'])
    n = len(df)
    print(f'  Feature pairs: {n}')

    # Merge edition metadata
    meta_cols = ['edition_id', 'book_id', 'author_id', 'publication_year',
                 'age_restriction', 'language_id', 'publisher_id', 'genres_cnt']
    df = df.merge(editions_df[meta_cols], on='edition_id', how='left')
    df = df.merge(embeddings_df, on='edition_id', how='left')
    for c in emb_c:
        df[c] = df[c].fillna(0.0)

    # Merge user demographics
    df = df.merge(users_df[['user_id', 'gender', 'age', 'age_cat']], on='user_id', how='left')
    df['gender'] = df['gender'].fillna(0).astype(int)
    df['age_num'] = df['age'].fillna(0).astype(float)
    df['age_cat'] = df['age_cat'].fillna(0).astype(int)
    df.drop(columns='age', inplace=True, errors='ignore')

    # ─── EASE score ───
    df['ease_score'] = 0.0
    for uid, group in df.groupby('user_id'):
        if uid not in ease_u2i: continue
        sc = ease_sc[ease_u2i[uid]]
        idx = group.index
        df.loc[idx, 'ease_score'] = [float(sc[ease_e2i[e]]) if e in ease_e2i else 0.0
                                      for e in group['edition_id'].values]

    # ─── ALS edition ───
    u_als = df['user_id'].map(als_u2i).fillna(-1).astype(int)
    i_als = df['edition_id'].map(als_i2i).fillna(-1).astype(int)
    m = (u_als != -1) & (i_als != -1)
    for col in ['als_dot', 'als_cos', 'als_user_norm', 'als_item_norm']:
        df[col] = 0.0
    if m.any():
        uv = als_uf[u_als[m].values]; iv = als_if[i_als[m].values]
        dots = np.einsum('ij,ij->i', uv, iv)
        un = np.linalg.norm(uv, axis=1); inn = np.linalg.norm(iv, axis=1)
        df.loc[m, 'als_dot'] = dots
        df.loc[m, 'als_cos'] = dots / (un * inn + 1e-9)
        df.loc[m, 'als_user_norm'] = un
        df.loc[m, 'als_item_norm'] = inn
    df['als_local_z'] = df.groupby('user_id')['als_dot'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0)

    # ─── ALS book / author / publisher domains ───
    for prefix, dom_col, dom_uf, dom_if, dom_u2i, dom_i2i in [
        ('als_book', 'book_id', als_b_uf, als_b_if, als_b_u2i, als_b_i2i),
        ('als_auth', 'author_id', als_a_uf, als_a_if, als_a_u2i, als_a_i2i),
        ('als_pub', 'publisher_id', als_p_uf, als_p_if, als_p_u2i, als_p_i2i),
    ]:
        du = df['user_id'].map(dom_u2i).fillna(-1).astype(int)
        di = df[dom_col].map(dom_i2i).fillna(-1).astype(int)
        m2 = (du != -1) & (di != -1)
        sc_arr = np.zeros(n, dtype=np.float32)
        if m2.any():
            sc_arr[m2] = np.einsum('ij,ij->i', dom_uf[du[m2].values], dom_if[di[m2].values])
        df[f'{prefix}_score'] = sc_arr
        df[f'{prefix}_zscore'] = df.groupby('user_id')[f'{prefix}_score'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0)

    # ─── SVD features (2 models) ───
    for tag, sue, sie, sk in [('svd1', svd_ue1, svd_ie1, svd_k1),
                               ('svd2', svd_ue2, svd_ie2, svd_k2)]:
        zero = np.zeros(sk)
        u_embs = np.array([sue.get(uid, zero) for uid in df['user_id'].values])
        i_embs = np.array([sie.get(eid, zero) for eid in df['edition_id'].values])
        dot = np.sum(u_embs * i_embs, axis=1)
        un = np.linalg.norm(u_embs, axis=1); inn = np.linalg.norm(i_embs, axis=1)
        cos = dot / (un * inn + 1e-9)
        df[f'{tag}_score'] = dot
        df[f'{tag}_cos'] = np.clip(cos, -1, 1)
        df[f'{tag}_angle'] = np.arccos(np.clip(cos, -1, 1))
        df[f'{tag}_u_norm'] = un
        df[f'{tag}_i_norm'] = inn
        df[f'{tag}_euc_dist'] = np.linalg.norm(u_embs - i_embs, axis=1)
        df[f'{tag}_proj_on_user'] = dot / (un + 1e-9)
        df[f'{tag}_item_orth_mag'] = np.sqrt(np.maximum(0, inn**2 - df[f'{tag}_proj_on_user']**2))
        # Element-wise (SVD1 only)
        if tag == 'svd1':
            n_ew = min(N_SVD_EW, sk)
            ew = u_embs[:, :n_ew] * i_embs[:, :n_ew]
            for j in range(n_ew):
                df[f'svd_ew_{j}'] = ew[:, j]

    # ─── ItemKNN score ───
    df['knn_score'] = 0.0
    for uid, group in df.groupby('user_id'):
        if uid not in ease_u2i: continue
        user_items = {ease_iids[j] for j in ease_mat[ease_u2i[uid]].indices}
        knn_scores = defaultdict(float)
        for e in user_items:
            if e in knn_nb:
                for ne, sim in knn_nb[e]:
                    knn_scores[ne] += sim
        df.loc[group.index, 'knn_score'] = [knn_scores.get(e, 0.0) for e in group['edition_id'].values]

    # (LightFM features removed -- EASE+ALS+SVD proven best combo)

    # ─── User stats ───
    user_stats = inter_df.groupby('user_id').agg(
        user_cnt=('edition_id', 'size'),
        user_read_cnt=('event_type', lambda x: (x == 2).sum()),
        user_wish_cnt=('event_type', lambda x: (x == 1).sum()),
        user_last_ts=('event_ts', 'max'),
        user_rating_mean=('rating', 'mean'),
        user_rating_cnt=('rating', lambda x: x.notna().sum()),
    ).reset_index()
    user_stats['user_read_share'] = user_stats['user_read_cnt'] / user_stats['user_cnt'].clip(1)
    user_stats['user_days_since'] = (last_ts - user_stats['user_last_ts']).dt.total_seconds() / 86400.0
    # User avg item popularity
    ipop = inter_df.groupby('edition_id').size().reset_index(name='_ipop')
    uip = inter_df.merge(ipop, on='edition_id', how='left')
    user_avg_pop = uip.groupby('user_id')['_ipop'].mean().reset_index(name='user_avg_item_pop')
    user_stats = user_stats.merge(user_avg_pop, on='user_id', how='left')
    user_stats.drop(columns='user_last_ts', inplace=True)
    df = df.merge(user_stats, on='user_id', how='left')
    for c in ['user_cnt', 'user_read_cnt', 'user_wish_cnt', 'user_rating_cnt']:
        df[c] = df[c].fillna(0).astype(int)
    df['user_read_share'] = df['user_read_share'].fillna(0)
    df['user_days_since'] = df['user_days_since'].fillna(9999)
    df['user_rating_mean'] = df['user_rating_mean'].fillna(0)
    df['user_avg_item_pop'] = df['user_avg_item_pop'].fillna(0)

    # ─── Item stats ───
    ic = inter_df.copy()
    ic['delta'] = (last_ts - ic['event_ts']).dt.total_seconds() / 86400.0
    ic['w7'] = np.exp(-ic['delta'] / 7.0)
    ic['w30'] = np.exp(-ic['delta'] / 30.0)
    item_stats = ic.groupby('edition_id').agg(
        item_pop=('event_type', 'size'),
        item_read_cnt=('event_type', lambda x: (x == 2).sum()),
        item_wish_cnt=('event_type', lambda x: (x == 1).sum()),
        item_last_ts=('event_ts', 'max'),
        item_decay_7=('w7', 'sum'), item_decay_30=('w30', 'sum'),
        item_rating_mean=('rating', 'mean'),
        item_rating_cnt=('rating', lambda x: x.notna().sum()),
    ).reset_index()
    item_stats['item_pop_log'] = np.log1p(item_stats['item_pop'])
    item_stats['item_read_ratio'] = item_stats['item_read_cnt'] / (item_stats['item_pop'] + 1e-9)
    item_stats['item_days_since'] = (last_ts - item_stats['item_last_ts']).dt.total_seconds() / 86400.0
    # Bayesian rating
    C = item_stats['item_rating_mean'].mean()
    item_stats['item_bayesian_rating'] = (
        (item_stats['item_rating_cnt'] * item_stats['item_rating_mean'] + 5 * C) /
        (item_stats['item_rating_cnt'] + 5))
    item_stats.drop(columns='item_last_ts', inplace=True)
    df = df.merge(item_stats, on='edition_id', how='left')
    for c in ['item_pop', 'item_read_cnt', 'item_wish_cnt', 'item_rating_cnt']:
        df[c] = df[c].fillna(0).astype(int)
    for c in ['item_pop_log', 'item_read_ratio', 'item_decay_7', 'item_decay_30',
              'item_days_since', 'item_rating_mean', 'item_bayesian_rating']:
        df[c] = df[c].fillna(0.0)

    # ─── Item pop windows (7d / 30d / 90d) ───
    ic_ed = ic.merge(editions_df[['edition_id', 'book_id', 'author_id']], on='edition_id', how='inner')
    for days, suffix in [(7, '7d'), (30, '30d'), (90, '90d')]:
        window = ic_ed[ic_ed['delta'] <= days]
        ipw = window.groupby('edition_id').size().rename(f'item_pop_{suffix}').reset_index()
        df = df.merge(ipw, on='edition_id', how='left')
        df[f'item_pop_{suffix}'] = df[f'item_pop_{suffix}'].fillna(0)
    df['item_pop_7_over_30'] = df['item_pop_7d'] / (df['item_pop_30d'] + 1e-9)
    df['item_pop_30_over_90'] = df['item_pop_30d'] / (df['item_pop_90d'] + 1e-9)
    df['item_pop_slope'] = df['item_pop_7_over_30'] - df['item_pop_30_over_90']

    # Book & author 30d popularity
    past30 = ic_ed[ic_ed['delta'] <= 30]
    for col, gcol in [('book_pop_30', 'book_id'), ('auth_pop_30', 'author_id')]:
        p30 = past30.groupby(gcol).size().rename(col).reset_index()
        df = df.merge(p30, on=gcol, how='left')
        df[col] = df[col].fillna(0)
        df[f'{col}_log'] = np.log1p(df[col])

    # ─── User time windows (7d / 30d) ───
    for days in [7, 30]:
        s = f'_{days}d'
        w = inter_df[inter_df['event_ts'] >= last_ts - pd.Timedelta(days=days)]
        if len(w) > 0:
            uw = w.groupby('user_id').agg(
                **{f'u_events{s}': ('event_type', 'count'),
                   f'u_reads{s}': ('event_type', lambda x: (x == 2).sum()),
                   f'u_wishes{s}': ('event_type', lambda x: (x == 1).sum())}
            ).reset_index()
            uw[f'u_read_ratio{s}'] = uw[f'u_reads{s}'] / (uw[f'u_events{s}'] + 1e-9)
        else:
            uw = pd.DataFrame(columns=['user_id', f'u_events{s}', f'u_reads{s}',
                                        f'u_wishes{s}', f'u_read_ratio{s}'])
        df = df.merge(uw, on='user_id', how='left')
        for c in [f'u_events{s}', f'u_reads{s}', f'u_wishes{s}', f'u_read_ratio{s}']:
            df[c] = df[c].fillna(0)
    df['u_events_7_over_30'] = df['u_events_7d'] / (df['u_events_30d'] + 1e-9)
    df['u_ratio_diff_7_30'] = df['u_read_ratio_7d'] - df['u_read_ratio_30d']

    # ─── Interaction / affinity features ───
    user_inter = inter_df.groupby(['user_id', 'edition_id']).size().rename('user_inter_cnt').reset_index()
    df = df.merge(user_inter, on=['user_id', 'edition_id'], how='left')
    df['user_inter_cnt'] = df['user_inter_cnt'].fillna(0).astype(int)

    inter_meta = inter_df.merge(editions_df[['edition_id', 'author_id', 'book_id', 'publisher_id']],
                                 on='edition_id', how='left')
    ua_cnt = inter_meta.groupby(['user_id', 'author_id']).size().rename('user_author_cnt').reset_index()
    ub_cnt = inter_meta.groupby(['user_id', 'book_id']).size().rename('user_book_cnt').reset_index()
    df = df.merge(ua_cnt, on=['user_id', 'author_id'], how='left')
    df = df.merge(ub_cnt, on=['user_id', 'book_id'], how='left')
    df['user_author_cnt'] = df['user_author_cnt'].fillna(0).astype(int)
    df['user_book_cnt'] = df['user_book_cnt'].fillna(0).astype(int)
    df['author_familiar'] = (df['user_author_cnt'] > 0).astype(int)

    # User book history
    bh = inter_meta.groupby(['user_id', 'book_id']).agg(
        ub_read=('event_type', lambda x: (x == 2).sum()),
        ub_wish=('event_type', lambda x: (x == 1).sum()),
        ub_rating=('rating', 'mean'),
    ).reset_index()
    bh['ub_any_read'] = (bh['ub_read'] > 0).astype(int)
    bh['ub_any_wish'] = (bh['ub_wish'] > 0).astype(int)
    bh['ub_max_signal'] = bh['ub_read'].clip(upper=1) * 3 + bh['ub_wish'].clip(upper=1) * (1 - bh['ub_read'].clip(upper=1))
    df = df.merge(bh, on=['user_id', 'book_id'], how='left')
    for c in ['ub_read', 'ub_wish', 'ub_any_read', 'ub_any_wish', 'ub_max_signal', 'ub_rating']:
        df[c] = df[c].fillna(0)

    # Author affinity (time-weighted + normalized)
    ic_auth = ic.merge(editions_df[['edition_id', 'author_id']], on='edition_id', how='left')
    ic_auth['tw'] = np.exp(-np.log(2) * ic_auth['delta'] / 30) * ic_auth['event_type'].map({1: 1.0, 2: 3.0}).fillna(0)
    auth_aff = ic_auth.groupby(['user_id', 'author_id'])['tw'].sum().reset_index(name='user_author_aff_tw')
    auth_pop_df = inter_meta.groupby('author_id').size().reset_index(name='author_pop')
    auth_aff = auth_aff.merge(auth_pop_df, on='author_id', how='left')
    auth_aff['user_author_aff_norm'] = auth_aff['user_author_aff_tw'] / (np.sqrt(auth_aff['author_pop'].fillna(0)) + 1e-6)
    df = df.merge(auth_aff[['user_id', 'author_id', 'user_author_aff_tw', 'user_author_aff_norm']],
                  on=['user_id', 'author_id'], how='left')
    df['user_author_aff_tw'] = df['user_author_aff_tw'].fillna(0)
    df['user_author_aff_norm'] = df['user_author_aff_norm'].fillna(0)

    # Publisher affinity (time-weighted)
    ic_pub = ic.merge(editions_df[['edition_id', 'publisher_id']], on='edition_id', how='left')
    ic_pub['tw'] = np.exp(-np.log(2) * ic_pub['delta'] / 30) * ic_pub['event_type'].map({1: 1.0, 2: 3.0}).fillna(0)
    pub_aff = ic_pub.groupby(['user_id', 'publisher_id'])['tw'].sum().reset_index(name='user_pub_aff_tw')
    df = df.merge(pub_aff, on=['user_id', 'publisher_id'], how='left')
    df['user_pub_aff_tw'] = df['user_pub_aff_tw'].fillna(0)

    # Author / publisher stats
    auth_stats = inter_meta.groupby('author_id').agg(
        author_pop=('event_type', 'count'),
        author_read_cnt=('event_type', lambda x: (x == 2).sum()),
        author_rating_mean=('rating', 'mean'),
    ).reset_index()
    auth_stats['author_read_ratio'] = auth_stats['author_read_cnt'] / (auth_stats['author_pop'] + 1e-9)
    pub_stats = inter_meta.groupby('publisher_id').agg(
        pub_pop=('event_type', 'count'),
        pub_rating_mean=('rating', 'mean'),
    ).reset_index()
    df = df.merge(auth_stats, on='author_id', how='left')
    df = df.merge(pub_stats, on='publisher_id', how='left')
    for c in ['author_pop', 'author_read_cnt', 'pub_pop']:
        df[c] = df[c].fillna(0)
    for c in ['author_rating_mean', 'author_read_ratio', 'pub_rating_mean']:
        df[c] = df[c].fillna(0)

    # ─── Language ───
    lang_inter = inter_df.merge(editions_df[['edition_id', 'language_id']], on='edition_id', how='left')
    ulang = lang_inter.groupby(['user_id', 'language_id']).size().reset_index(name='cnt')
    ulang = ulang.sort_values(['user_id', 'cnt'], ascending=[True, False])
    utop = ulang.drop_duplicates('user_id')[['user_id', 'language_id']].rename(columns={'language_id': 'user_top_lang'})
    df = df.merge(utop, on='user_id', how='left')
    df['user_top_lang'] = df['user_top_lang'].fillna(-1).astype(int)
    df['lang_match'] = (df['language_id'] == df['user_top_lang']).astype(int)

    # ─── Genre matching ───
    ed2genre = editions_df[['edition_id', 'book_id']].merge(book_genres_df, on='book_id', how='left')
    past_genres = inter_df[['user_id', 'edition_id', 'event_type']].merge(ed2genre, on='edition_id', how='left')
    past_genres['weight'] = np.where(past_genres['event_type'] == 2, 2.0, 1.0)
    ug_weight = past_genres.groupby(['user_id', 'genre_id'])['weight'].sum().reset_index()
    ug_weight['wr'] = ug_weight.groupby('user_id')['weight'].rank(method='first', ascending=False)
    ug_top = ug_weight[ug_weight['wr'] <= 10].drop(columns='wr')
    pair_genres = df[['user_id', 'edition_id']].merge(ed2genre, on='edition_id', how='left')
    pg_match = pair_genres.merge(ug_top, on=['user_id', 'genre_id'], how='inner')
    pg_agg = pg_match.groupby(['user_id', 'edition_id']).agg(
        genre_match_cnt=('genre_id', 'nunique'),
        genre_match_wsum=('weight', 'sum')
    ).reset_index()
    df = df.merge(pg_agg, on=['user_id', 'edition_id'], how='left')
    df['genre_match_cnt'] = df['genre_match_cnt'].fillna(0).astype(int)
    df['genre_match_wsum'] = df['genre_match_wsum'].fillna(0)
    df['genre_match_frac'] = df['genre_match_cnt'] / df['genres_cnt'].clip(1)

    # Genre entropy (user + item)
    ui_genres = inter_df[['user_id', 'edition_id']].merge(ed2genre[['edition_id', 'genre_id']], on='edition_id', how='left').dropna()
    ug_cnt = ui_genres.groupby(['user_id', 'genre_id']).size().reset_index(name='gcnt')
    ug_tot = ug_cnt.groupby('user_id')['gcnt'].sum().reset_index(name='gtot')
    ug_cnt = ug_cnt.merge(ug_tot, on='user_id')
    ug_cnt['p'] = ug_cnt['gcnt'] / (ug_cnt['gtot'] + 1e-9)
    user_entropy = ug_cnt.groupby('user_id').apply(lambda d: -(d['p'] * np.log(d['p'] + 1e-9)).sum()).reset_index(name='user_genre_entropy')
    df = df.merge(user_entropy, on='user_id', how='left')
    df['user_genre_entropy'] = df['user_genre_entropy'].fillna(0)

    ig_cnt = ed2genre.dropna(subset=['genre_id']).groupby(['edition_id', 'genre_id']).size().reset_index(name='gcnt')
    ig_tot = ig_cnt.groupby('edition_id')['gcnt'].sum().reset_index(name='gtot')
    ig_cnt = ig_cnt.merge(ig_tot, on='edition_id')
    ig_cnt['p'] = ig_cnt['gcnt'] / (ig_cnt['gtot'] + 1e-9)
    item_entropy = ig_cnt.groupby('edition_id').apply(lambda d: -(d['p'] * np.log(d['p'] + 1e-9)).sum()).reset_index(name='item_genre_entropy')
    df = df.merge(item_entropy, on='edition_id', how='left')
    df['item_genre_entropy'] = df['item_genre_entropy'].fillna(0)

    # User genre diversity + author diversity
    user_genre_div = ui_genres.groupby('user_id')['genre_id'].nunique().reset_index(name='user_genre_diversity')
    df = df.merge(user_genre_div, on='user_id', how='left')
    df['user_genre_diversity'] = df['user_genre_diversity'].fillna(0)
    user_auth_div = inter_meta.groupby('user_id')['author_id'].nunique().reset_index(name='user_author_diversity')
    df = df.merge(user_auth_div, on='user_id', how='left')
    df['user_author_diversity'] = df['user_author_diversity'].fillna(0)

    # ─── Text profile features ───
    past_emb = inter_df[['user_id', 'edition_id', 'event_type', 'event_ts']].merge(
        embeddings_df[['edition_id'] + emb_c], on='edition_id', how='inner')
    delta = (last_ts - past_emb['event_ts']).dt.total_seconds() / 86400.0
    past_emb['weight'] = np.exp(-delta / 60.0) * np.where(past_emb['event_type'] == 2, 2.0, 1.0)
    w_embs = past_emb[emb_c].multiply(past_emb['weight'], axis=0)
    u_sum = w_embs.assign(user_id=past_emb['user_id']).groupby('user_id')[emb_c].sum()
    w_sum = past_emb.groupby('user_id')['weight'].sum()
    u_vecs = u_sum.div(w_sum, axis=0).fillna(0.0)
    u_norms = np.linalg.norm(u_vecs.values, axis=1, keepdims=True)
    u_vecs = u_vecs.div(u_norms + 1e-9)
    u_vecs.columns = [f'u_{c}' for c in emb_c]
    df = df.merge(u_vecs, on='user_id', how='left')
    for c in emb_c:
        df[f'u_{c}'] = df[f'u_{c}'].fillna(0.0)
    u_mat = df[[f'u_{c}' for c in emb_c]].values
    i_mat = df[emb_c].values
    dot = np.einsum('ij,ij->i', u_mat, i_mat)
    i_norm = np.linalg.norm(i_mat, axis=1); u_norm = np.linalg.norm(u_mat, axis=1)
    df['text_cos'] = dot / (i_norm * u_norm + 1e-9)
    df['text_l2'] = np.linalg.norm(u_mat - i_mat, axis=1)
    df['text_loc_z'] = df.groupby('user_id')['text_cos'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)).fillna(0)

    # Author text cosine
    auth_emb = editions_df[['edition_id', 'author_id']].merge(
        embeddings_df[['edition_id'] + emb_c], on='edition_id', how='inner')
    a_vecs = auth_emb.groupby('author_id')[emb_c].mean()
    a_norms = np.linalg.norm(a_vecs.values, axis=1, keepdims=True)
    a_vecs = a_vecs.div(a_norms + 1e-9)
    a_vecs.columns = [f'a_{c}' for c in emb_c]
    df = df.merge(a_vecs, on='author_id', how='left')
    for c in emb_c:
        df[f'a_{c}'] = df[f'a_{c}'].fillna(0.0)
    a_mat = df[[f'a_{c}' for c in emb_c]].values
    df['text_auth_cos'] = np.einsum('ij,ij->i', u_mat, a_mat)
    drop_emb = emb_c + [f'u_{c}' for c in emb_c] + [f'a_{c}' for c in emb_c]
    df.drop(columns=drop_emb, inplace=True)

    # ─── Advanced features ───
    # Novelty preference
    w30_data = inter_df[inter_df['event_ts'] >= last_ts - pd.Timedelta(days=30)]
    if len(w30_data) > 0:
        wp = w30_data.groupby('edition_id').size().reset_index(name='pop_w')
        q = wp['pop_w'].quantile(0.2)
        low_items = set(wp[wp['pop_w'] <= q]['edition_id'])
        w30u = w30_data.copy()
        w30u['is_low'] = w30u['edition_id'].isin(low_items).astype(int)
        u_novelty = w30u.groupby('user_id')['is_low'].mean().reset_index(name='user_novelty_pref')
    else:
        u_novelty = pd.DataFrame(columns=['user_id', 'user_novelty_pref'])
    df = df.merge(u_novelty, on='user_id', how='left')
    df['user_novelty_pref'] = df['user_novelty_pref'].fillna(0)

    # Popularity preference
    if 'item_pop' in item_stats.columns:
        ip_r = item_stats[['edition_id', 'item_pop']].copy()
        ip_r['item_pop_rank_pct'] = ip_r['item_pop'].rank(pct=True)
        uip2 = inter_df.merge(ip_r[['edition_id', 'item_pop_rank_pct']], on='edition_id', how='left')
        uip2['item_pop_rank_pct'] = uip2['item_pop_rank_pct'].fillna(0.5)
        u_pop_pref = uip2.groupby('user_id')['item_pop_rank_pct'].mean().reset_index(name='user_pop_pref')
        df = df.merge(u_pop_pref, on='user_id', how='left')
    df['user_pop_pref'] = df.get('user_pop_pref', pd.Series(0.5, index=df.index)).fillna(0.5)

    # Rating diff, pop ratios, age
    df['rating_diff'] = np.abs(df['user_rating_mean'].fillna(8.3) - df['item_rating_mean'].fillna(8.3))
    df['item_pop_vs_user'] = df['item_pop'] / (df['user_avg_item_pop'].fillna(1) + 1e-9)
    df['item_vs_author_pop'] = df['item_pop'] / (df['author_pop'].fillna(1) + 1e-9)
    df['pub_year'] = df['publication_year'].astype(float).fillna(0)
    df.loc[df['pub_year'] < 1900, 'pub_year'] = 2016.0
    df['year_old'] = (2025.0 - df['pub_year']).clip(0, 100)
    df['pub_recent'] = (df['pub_year'] >= 2020).astype(int)
    df['age_restr'] = df['age_restriction'].astype(float).fillna(0)
    df['is_age_ok'] = ((df['age_num'] == 0) | (df['age_num'] >= df['age_restr'])).astype(int)

    # ─── November Echo features ───
    if echo_start is not None and echo_end is not None:
        # Incident month = month before echo
        inc_start = echo_start - pd.DateOffset(months=1)
        inc_end = echo_start

        echo_inter = inter_df[(inter_df['event_ts'] >= echo_start) &
                               (inter_df['event_ts'] < echo_end)]
        echo_meta = echo_inter.merge(
            editions_df[['edition_id', 'author_id', 'book_id']], on='edition_id', how='left')
        inc_inter = inter_df[(inter_df['event_ts'] >= inc_start) &
                              (inter_df['event_ts'] < inc_end)]
        inc_meta = inc_inter.merge(
            editions_df[['edition_id', 'author_id', 'book_id']], on='edition_id', how='left')

        # Author echo
        echo_ua = echo_meta.groupby(['user_id', 'author_id']).size().reset_index(name='nov_author_cnt')
        inc_ua = inc_meta.groupby(['user_id', 'author_id']).size().reset_index(name='oct_author_cnt')
        df = df.merge(echo_ua, on=['user_id', 'author_id'], how='left')
        df = df.merge(inc_ua, on=['user_id', 'author_id'], how='left')
        df['nov_author_cnt'] = df['nov_author_cnt'].fillna(0)
        df['oct_author_cnt'] = df['oct_author_cnt'].fillna(0)
        df['author_echo_gap'] = df['nov_author_cnt'] - df['oct_author_cnt']
        df['author_echo_ratio'] = df['nov_author_cnt'] / (df['oct_author_cnt'] + 1)

        # Exponential decay within echo month (15-day half-life)
        if len(echo_inter) > 0:
            echo_last = echo_inter['event_ts'].max()
            echo_meta_c = echo_meta.copy()
            echo_meta_c['delta'] = (echo_last - echo_meta_c['event_ts']).dt.total_seconds() / 86400.0
            echo_meta_c['decay_w'] = np.exp(-echo_meta_c['delta'] / 15.0)
            echo_decay = echo_meta_c.groupby(['user_id', 'author_id'])['decay_w'].sum().reset_index(
                name='nov_author_decay')
            df = df.merge(echo_decay, on=['user_id', 'author_id'], how='left')
            df['nov_author_decay'] = df['nov_author_decay'].fillna(0)
        else:
            df['nov_author_decay'] = 0.0

        # Book echo
        echo_ub = echo_meta.groupby(['user_id', 'book_id']).size().reset_index(name='echo_book_cnt')
        inc_ub = inc_meta.groupby(['user_id', 'book_id']).size().reset_index(name='_inc_book_cnt')
        df = df.merge(echo_ub, on=['user_id', 'book_id'], how='left')
        df = df.merge(inc_ub, on=['user_id', 'book_id'], how='left')
        df['echo_book_cnt'] = df['echo_book_cnt'].fillna(0)
        df['echo_book_gap'] = df['echo_book_cnt'] - df['_inc_book_cnt'].fillna(0)
        df.drop(columns='_inc_book_cnt', inplace=True, errors='ignore')

        print(f'  Echo features added (echo: {echo_start.date()} to {echo_end.date()})')
    else:
        for c in ['nov_author_cnt', 'oct_author_cnt', 'author_echo_gap',
                  'author_echo_ratio', 'nov_author_decay', 'echo_book_cnt', 'echo_book_gap']:
            df[c] = 0.0

    # ─── Per-user rank & z-score features ───
    rank_cols = ['ease_score', 'als_dot', 'knn_score', 'svd1_score', 'svd2_score',
                 'text_cos', 'text_auth_cos', 'item_pop_log', 'item_decay_30',
                 'auth_pop_30_log', 'publication_year', 'author_echo_gap']
    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_g_rank'] = df.groupby('user_id')[col].rank(pct=True, method='average')

    z_cols = ['item_pop_log', 'svd1_score', 'svd2_score', 'text_cos', 'genre_match_wsum',
              'item_bayesian_rating', 'user_author_aff_norm', 'user_pop_pref',
              'author_echo_gap', 'nov_author_decay']
    for col in z_cols:
        if col in df.columns:
            mean = df.groupby('user_id')[col].transform('mean')
            std = df.groupby('user_id')[col].transform('std').replace(0, np.nan)
            df[f'z_{col}'] = ((df[col] - mean) / (std + 1e-9)).fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════
# TRAINING (Monthly folds Jul-Oct + neg-ratio search)  [CACHED]
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60, "\nTRAINING\n", "=" * 60)

# 2-month folds: (incident_month, echo_month)
TRAIN_FOLDS = [(7, 8), (8, 9), (9, 10), (10, 11)]
FOLD_DECAY = 0.3  # exponential decay for older folds
cat_cols = ['user_id', 'edition_id', 'book_id', 'author_id', 'language_id',
            'publisher_id', 'age_restriction', 'user_top_lang', 'gender', 'age_cat']


def downsample_negatives(df, neg_ratio, hard_frac=HARD_NEG_FRAC, seed=42):
    """Ratio-based neg sampling: neg_ratio negatives per positive, hard_frac from top EASE."""
    keep_idx = []
    for uid, group in df.groupby('user_id'):
        pos = group[group['target'] > 0]
        neg = group[group['target'] == 0]
        keep_idx.extend(pos.index.tolist())
        n_pos = len(pos)
        if n_pos == 0 or len(neg) == 0:
            continue
        n_neg_want = min(len(neg), n_pos * neg_ratio)
        if n_neg_want >= len(neg):
            keep_idx.extend(neg.index.tolist())
        else:
            n_hard = int(n_neg_want * hard_frac)
            n_rand = n_neg_want - n_hard
            if n_hard > 0 and 'ease_score' in neg.columns:
                hard = neg.nlargest(min(n_hard, len(neg)), 'ease_score')
                keep_idx.extend(hard.index.tolist())
                remaining = neg.drop(hard.index)
            else:
                remaining = neg
                n_rand = n_neg_want
            if n_rand > 0 and len(remaining) > 0:
                sampled = remaining.sample(n=min(n_rand, len(remaining)),
                                           random_state=seed)
                keep_idx.extend(sampled.index.tolist())
    return df.loc[keep_idx].sort_values('user_id').reset_index(drop=True)


def compute_ndcg_at_k(true_arr, pred_arr, group_ids, k=20):
    """Compute mean NDCG@K grouped by group_ids."""
    df_tmp = pd.DataFrame({'g': group_ids, 't': true_arr, 'p': pred_arr})
    ndcgs = []
    for _, grp in df_tmp.groupby('g'):
        if grp['t'].sum() == 0:
            continue
        order = np.argsort(-grp['p'].values)
        rel = grp['t'].values[order][:k].astype(np.float64)
        gains = (2.0**rel - 1.0) / np.log2(np.arange(2, len(rel) + 2)).astype(np.float64)
        dcg = gains.sum()
        ideal = np.sort(grp['t'].values)[::-1][:k].astype(np.float64)
        igains = (2.0**ideal - 1.0) / np.log2(np.arange(2, len(ideal) + 2)).astype(np.float64)
        idcg = igains.sum()
        ndcgs.append(dcg / (idcg + 1e-10))
    return np.mean(ndcgs) if ndcgs else 0.0


# ── Check cache for final model ──
models = []
feat_cols = None

_cached_model = load_cache('catboost_final')
_cached_fc = load_cache('feat_cols')

if _cached_model is not None and _cached_fc is not None:
    models = [_cached_model]
    feat_cols = _cached_fc
    _best_ratio = load_cache('best_neg_ratio')
    print(f'  Final model loaded from cache (neg_ratio={_best_ratio})')
else:
    # ── Step 1: Build per-fold feature DataFrames ──
    fold_dfs = []
    for month, echo_month in TRAIN_FOLDS:
        print(f'\n--- Fold: month={month}, echo={echo_month} ---')

        _fold_cache = load_cache(f'fold_df_{month}_{echo_month}')
        if _fold_cache is not None:
            fold_df = _fold_cache
            if feat_cols is None:
                feat_cols = [c for c in fold_df.columns if c not in ['target', 'fold', 'fold_echo']]
                print(f'  Feature count: {len(feat_cols)}')
            print(f'  Fold DF (cached): {fold_df.shape}')
            fold_dfs.append(fold_df)
            continue

        # Month boundaries
        month_start = pd.Timestamp(f'2025-{month:02d}-01')
        month_end = pd.Timestamp(f'2025-{month + 1:02d}-01') if month < 12 \
            else pd.Timestamp('2026-01-01')
        echo_start = pd.Timestamp(f'2025-{echo_month:02d}-01')
        echo_end = pd.Timestamp(f'2025-{echo_month + 1:02d}-01') if echo_month < 12 \
            else pd.Timestamp('2026-01-01')

        month_inter = interactions[
            (interactions.event_ts >= month_start) & (interactions.event_ts < month_end)]
        month_tgt = month_inter[month_inter.user_id.isin(target_set)]
        # Track (user, edition) pairs in this month
        month_pairs = month_tgt.groupby('user_id')['edition_id'].apply(set).to_dict()

        # Remove ~20%  (removed = {uid: set(eid, ...)})
        rng = np.random.RandomState(42 + month)
        removed = {}
        for uid, items_set in month_pairs.items():
            items_list = list(items_set)
            if len(items_list) == 0:
                continue
            n = max(1, int(len(items_list) * 0.2))
            idx = rng.choice(len(items_list), n, replace=False)
            removed[uid] = {items_list[i] for i in idx}

        if not removed:
            print(f'  No removable interactions in month {month}, skipping')
            continue

        # Build visible data = all interactions minus removed pairs
        removed_pairs = pd.DataFrame(
            [(u, e) for u, eids in removed.items() for e in eids],
            columns=['user_id', 'edition_id'])
        removed_pairs['_rm'] = 1
        visible = interactions.merge(removed_pairs, on=['user_id', 'edition_id'], how='left')
        visible = visible[visible['_rm'].isna()].drop(columns='_rm')

        print(f'  Removed {len(removed_pairs)} pairs from {len(removed)} users')
        print(f'  Visible interactions: {len(visible)}')

        # Candidates
        cands, ease_d, als_d, svd1u, svd1i, svd1k, knn_d = \
            generate_all_candidates(
                list(removed.keys()), visible, hack, editions,
                embeddings, book_genres, month_start,
                echo_start=echo_start, echo_end=echo_end)

        # Domain models for features
        print('  Training domain models...')
        als_book_d = train_als_domain(visible, editions, 'book_id')
        als_auth_d = train_als_domain(visible, editions, 'author_id')
        als_pub_d = train_als_domain(visible, editions, 'publisher_id')
        svd2u, svd2i, svd2k = train_svd_cf(visible, n_factors=SVD2_FACTORS, use_log=False)

        # Inject positives into candidates
        for uid, rem in removed.items():
            if uid in cands:
                cands[uid] |= rem

        # Build features
        print('  Building features...')
        t0 = time.time()
        fold_df = build_features_df(
            cands, visible, editions, users, book_genres, embeddings,
            ease_d, als_d, als_book_d, als_auth_d, als_pub_d,
            svd1u, svd1i, svd1k, svd2u, svd2i, svd2k,
            knn_d,
            echo_start=echo_start, echo_end=echo_end)
        print(f'  Features: {fold_df.shape}, {time.time()-t0:.0f}s')

        # Binary labels: 1=positive (read or wish), 0=negative
        fold_df['target'] = 0
        rem_label = pd.DataFrame(
            [(u, e) for u, eids in removed.items() for e in eids],
            columns=['user_id', 'edition_id'])
        rem_label['_pos'] = 1
        fold_df = fold_df.merge(rem_label, on=['user_id', 'edition_id'], how='left')
        fold_df['target'] = fold_df['_pos'].fillna(0).astype(int)
        fold_df.drop(columns='_pos', inplace=True)
        fold_df['fold'] = month
        fold_df['fold_echo'] = echo_month

        n_pos = (fold_df['target'] == 1).sum()
        n_neg = (fold_df['target'] == 0).sum()
        print(f'  Labels: {n_pos} positive, {n_neg} negative')

        if feat_cols is None:
            feat_cols = [c for c in fold_df.columns if c not in ['target', 'fold', 'fold_echo']]
            print(f'  Feature count: {len(feat_cols)}')

        save_cache(f'fold_df_{month}_{echo_month}', fold_df)
        fold_dfs.append(fold_df)

        del ease_d, als_d, als_book_d, als_auth_d, als_pub_d
        del svd1u, svd1i, svd2u, svd2i, knn_d
        gc.collect()

    # Pool all folds
    all_train = pd.concat(fold_dfs, ignore_index=True)
    del fold_dfs; gc.collect()
    print(f'\nPooled training: {all_train.shape}')
    print(f'  Positives: {(all_train["target"] > 0).sum()}')
    print(f'  Negatives: {(all_train["target"] == 0).sum()}')

    # ── Step 2: Neg-ratio search (val=October, train=Jul+Aug+Sep) ──
    print('\n--- Neg-ratio search ---')
    val_mask = all_train['fold'] == 10
    val_df = all_train[val_mask].copy()
    search_train_base = all_train[~val_mask].copy()

    # Prepare val
    for c in cat_cols:
        if c in val_df.columns:
            val_df[c] = val_df[c].fillna(-1).astype(int)

    best_ratio = NEG_RATIOS[0]
    best_ndcg = -1.0

    for ratio in NEG_RATIOS:
        print(f'\n  Trying neg_ratio={ratio}...')
        ds = downsample_negatives(search_train_base, neg_ratio=ratio, seed=42)

        # Keep only users with positives
        umax = ds.groupby('user_id')['target'].max()
        ds = ds[ds['user_id'].isin(umax[umax > 0].index)]

        for c in cat_cols:
            if c in ds.columns:
                ds[c] = ds[c].fillna(-1).astype(int)

        print(f'    Train: {len(ds)} rows, {ds["user_id"].nunique()} users')

        pool_tr = Pool(
            ds[feat_cols], label=ds['target'],
            group_id=ds['user_id'].astype(str),
            cat_features=[c for c in cat_cols if c in feat_cols])

        mdl = CatBoostRanker(
            loss_function='YetiRank', eval_metric='NDCG:top=20;type=Exp',
            depth=CB_DEPTH, iterations=3000, learning_rate=CB_LR,
            l2_leaf_reg=CB_L2, random_strength=2.0,
            bootstrap_type='Bernoulli', subsample=0.8,
            od_type='Iter', od_wait=200,
            random_seed=42, verbose=1000,
            task_type='GPU' if USE_GPU else 'CPU',
            **(dict(gpu_ram_part=0.8) if USE_GPU else dict(thread_count=-1)),
            allow_writing_files=False)
        mdl.fit(pool_tr)

        pool_val = Pool(
            val_df[feat_cols],
            group_id=val_df['user_id'].astype(str),
            cat_features=[c for c in cat_cols if c in feat_cols])
        preds = mdl.predict(pool_val)

        ndcg = compute_ndcg_at_k(
            val_df['target'].values, preds,
            val_df['user_id'].values, k=20)
        print(f'    NDCG@20 = {ndcg:.5f}')

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_ratio = ratio

        del ds, pool_tr, pool_val, mdl
        gc.collect()

    print(f'\n  ▶ Best neg_ratio: {best_ratio} (NDCG@20={best_ndcg:.5f})')
    save_cache('best_neg_ratio', best_ratio)

    # ── Step 3: Final model on ALL 4 folds with best ratio + fold decay ──
    print(f'\n--- Final training (neg_ratio={best_ratio}, all 4 folds, fold decay) ---')
    final_train = downsample_negatives(all_train, neg_ratio=best_ratio, seed=42)

    umax = final_train.groupby('user_id')['target'].max()
    final_train = final_train[final_train['user_id'].isin(umax[umax > 0].index)]

    for c in cat_cols:
        if c in final_train.columns:
            final_train[c] = final_train[c].fillna(-1).astype(int)

    # Exponential decay weighting: recent folds weighted higher
    latest_echo = max(em for _, em in TRAIN_FOLDS)
    final_train['sample_weight'] = np.exp(-FOLD_DECAY * (latest_echo - final_train['fold_echo']))
    final_train['sample_weight'] /= final_train['sample_weight'].mean()  # normalize mean=1

    print(f'  Final train: {len(final_train)} rows, '
          f'{final_train["user_id"].nunique()} users, '
          f'{(final_train["target"] > 0).sum()} pos')
    print(f'  Fold weights: {dict(final_train.groupby("fold")["sample_weight"].mean().round(3))}')

    pool_final = Pool(
        final_train[feat_cols], label=final_train['target'],
        group_id=final_train['user_id'].astype(str),
        cat_features=[c for c in cat_cols if c in feat_cols],
        weight=final_train['sample_weight'].values)

    model = CatBoostRanker(
        loss_function='YetiRank', eval_metric='NDCG:top=20;type=Exp',
        depth=CB_DEPTH, iterations=CB_ITER, learning_rate=CB_LR,
        l2_leaf_reg=CB_L2, random_strength=2.0,
        bootstrap_type='Bernoulli', subsample=0.8,
        od_type='Iter', od_wait=CB_OD_WAIT,
        random_seed=42, verbose=500,
        task_type='GPU' if USE_GPU else 'CPU',
        **(dict(gpu_ram_part=0.8) if USE_GPU else dict(thread_count=-1)),
        allow_writing_files=False)
    model.fit(pool_final)
    models = [model]

    # Feature importance
    fi = model.get_feature_importance(pool_final)
    fi_sorted = sorted(zip(feat_cols, fi), key=lambda x: -x[1])
    print('\n  Top 20 features:')
    for fn, fv in fi_sorted[:20]:
        print(f'    {fn:>35}: {fv:.1f}')

    save_cache('catboost_final', model)
    save_cache('feat_cols', feat_cols)

    del final_train, pool_final, all_train
    gc.collect()

print(f'\nTrained {len(models)} model(s)')


# ═══════════════════════════════════════════════════════════════
# INFERENCE (October — full data)  [CACHED]
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60, "\nINFERENCE\n", "=" * 60)

_test_cache = load_cache('test_df')
if _test_cache is not None:
    test_df = _test_cache
    print(f'Inference features (cached): {test_df.shape}')
else:
    # November is echo month for October incident
    ECHO_START = pd.Timestamp('2025-11-01')
    ECHO_END   = pd.Timestamp('2025-12-01')

    # Generate candidates using ALL data
    inf_cands, inf_ease, inf_als, inf_svd1u, inf_svd1i, inf_svd1k, inf_knn = \
        generate_all_candidates(target_users, interactions, hack, editions,
                                 embeddings, book_genres, INCIDENT_START,
                                 echo_start=ECHO_START, echo_end=ECHO_END)

    # Train domain models for features on ALL data
    print('Training domain models for inference features...')
    inf_als_book = train_als_domain(interactions, editions, 'book_id')
    inf_als_auth = train_als_domain(interactions, editions, 'author_id')
    inf_als_pub = train_als_domain(interactions, editions, 'publisher_id')
    inf_svd2u, inf_svd2i, inf_svd2k = train_svd_cf(interactions, n_factors=SVD2_FACTORS, use_log=False)

    # Build features
    print('\nBuilding inference features...')
    t0 = time.time()
    test_df = build_features_df(
        inf_cands, interactions, editions, users, book_genres, embeddings,
        inf_ease, inf_als, inf_als_book, inf_als_auth, inf_als_pub,
        inf_svd1u, inf_svd1i, inf_svd1k, inf_svd2u, inf_svd2i, inf_svd2k,
        inf_knn,
        echo_start=ECHO_START, echo_end=ECHO_END)
    print(f'Inference features: {test_df.shape}, {time.time()-t0:.0f}s')

    save_cache('test_df', test_df)

# Align columns
for c in feat_cols:
    if c not in test_df.columns:
        test_df[c] = 0
for c in cat_cols:
    if c in test_df.columns:
        test_df[c] = test_df[c].fillna(-1).astype(int)

# Predict with all models and average
print('\nPredicting with ensemble...')
all_preds = []
for i, mdl in enumerate(models):
    test_pool = Pool(
        test_df[feat_cols],
        group_id=test_df['user_id'].astype(str),
        cat_features=[c for c in cat_cols if c in feat_cols]
    )
    preds = mdl.predict(test_pool)
    all_preds.append(preds)
    print(f'  Model {i+1}: done')

# Weighted average (more recent windows get higher weight)
weights = [1.0, 0.8, 0.6]  # window 1 (most recent) gets most weight
w_total = sum(weights[:len(models)])
test_df['score'] = sum(p * w for p, w in zip(all_preds, weights)) / w_total


# ═══════════════════════════════════════════════════════════════
# POST-PROCESSING + SUBMISSION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60, "\nPOST-PROCESSING\n", "=" * 60)

# Observed interactions
all_obs = defaultdict(set)
for u, e in zip(interactions['user_id'], interactions['edition_id']):
    all_obs[u].add(e)

# User books for dedup
user_books = defaultdict(set)
for u in target_set:
    for e in all_obs.get(u, set()):
        if e in ed_info.index:
            user_books[u].add(ed_info.loc[e, 'book_id'])

# Incident pairs to exclude
incident_inter = interactions[(interactions.event_ts >= INCIDENT_START) & (interactions.event_ts < INCIDENT_END)]
incident_pairs = set(zip(incident_inter['user_id'], incident_inter['edition_id']))

# Popular fallback
pop_items = incident_inter[incident_inter.user_id.isin(target_set)].groupby('edition_id').size()
pop_items = pop_items.sort_values(ascending=False).index.tolist()[:200]

# Sort and filter
test_df = test_df.sort_values(['user_id', 'score'], ascending=[True, False])
results = []

for uid in target_users:
    user_df = test_df[test_df['user_id'] == uid]
    obs = all_obs.get(uid, set())
    ubids = user_books.get(uid, set())

    filtered = []
    seen_books = set()
    for _, row in user_df.iterrows():
        eid = row['edition_id']
        if eid in obs:
            continue
        if (uid, eid) in incident_pairs:
            continue
        if eid in ed_info.index:
            if ed_info.loc[eid, 'language_id'] != 119:
                continue
            bid = ed_info.loc[eid, 'book_id']
            if bid in ubids or bid in seen_books:
                continue
            seen_books.add(bid)
        filtered.append(eid)
        if len(filtered) >= 20:
            break

    # Pad with popular items
    if len(filtered) < 20:
        existing = set(filtered)
        for eid in pop_items:
            if eid not in existing and eid not in obs:
                if eid in ed_info.index and ed_info.loc[eid, 'language_id'] == 119:
                    bid = ed_info.loc[eid, 'book_id']
                    if bid not in ubids and bid not in seen_books:
                        filtered.append(eid)
                        seen_books.add(bid)
                        existing.add(eid)
                        if len(filtered) >= 20:
                            break

    for rank, eid in enumerate(filtered[:20], 1):
        results.append({'user_id': uid, 'edition_id': eid, 'rank': rank})

sub = pd.DataFrame(results)
print(f'\nSubmission: {sub.shape}')
print(f'Users: {sub["user_id"].nunique()}')
per_user = sub.groupby('user_id').size()
print(f'Rows/user: min={per_user.min()}, max={per_user.max()}')

assert sub['user_id'].nunique() == len(target_users), f"Missing users! {sub['user_id'].nunique()}"
assert all(per_user == 20), "Not all 20 per user!"

sub.to_csv('sub_v13.csv', index=False)
print('\n=== SAVED sub_v13.csv ===')



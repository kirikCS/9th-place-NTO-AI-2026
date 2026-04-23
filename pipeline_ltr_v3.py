"""
LTR V3.1: Multi-source candidates + CatBoost YetiRank + Echo features
======================================================================
- 5 candidate sources: EASE, ALS, SVD, ItemKNN, Author (no TFIDFRec)
- Per-fold model training (no shared model leakage)
- SVD element-wise products as dense features for CatBoost
- Echo features: next-month signals (author, genre, SBERT, same-book)
- Graded labels: neg=0, wish=1, read=2
- Train: July + August + September, Validate: October

Required:  pip install catboost implicit scikit-learn scipy pandas numpy
Optional:  pip install sentence-transformers torch  (for SBERT features)
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors
from catboost import CatBoostRanker, Pool
import warnings, sys, time, gc, os, pickle

warnings.filterwarnings('ignore')

HAS_IMPLICIT = False
try:
    from implicit.als import AlternatingLeastSquares
    HAS_IMPLICIT = True
except ImportError:
    pass

# ======================== CONFIG ========================
DATA_DIR = os.environ.get('DATA_DIR', 'data')
DATA_ENRICHED_DIR = os.environ.get('DATA_ENRICHED_DIR', 'data_enriched')
SBERT_CACHE = 'cache_v2/sbert_embs.pkl'
USE_GPU = True
SVD_K = 48
ALS_FACTORS = 128
ALS_ITERS = 30
N_CAND_EASE = 250
N_CAND_ALS = 100
N_CAND_SVD = 100
N_CAND_KNN = 50
NEG_PER_USER = 150

def log(*a):
    sys.stdout.write(' '.join(map(str, a)) + '\n')
    sys.stdout.flush()

def ndcg_at_k(rec, rel, k=20):
    if not rel:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, e in enumerate(rec[:k]) if e in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mean_ndcg(preds, gt, k=20):
    s = [ndcg_at_k(preds.get(u, []), gt.get(u, set()), k) for u in gt]
    return float(np.mean(s)) if s else 0.0


# ======================== MODEL FUNCTIONS ========================

def build_matrix(pairs_df):
    uids = sorted(pairs_df['user_id'].unique())
    iids = sorted(pairs_df['edition_id'].unique())
    u2i = {u: i for i, u in enumerate(uids)}
    i2i = {e: i for i, e in enumerate(iids)}
    r = pairs_df['user_id'].map(u2i).values
    c = pairs_df['edition_id'].map(i2i).values
    mat = csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                     shape=(len(uids), len(iids)))
    return mat, uids, iids, u2i, i2i


def build_weighted_matrix(inter_df, u2i, i2i, nu, ni):
    """Weighted matrix for SVD: wish=1, read=3, log1p."""
    df = inter_df[inter_df['user_id'].isin(u2i) & inter_df['edition_id'].isin(i2i)].copy()
    df['w'] = df['event_type'].map({1: 1.0, 2: 3.0}).fillna(0.5)
    agg = df.groupby(['user_id', 'edition_id'])['w'].sum().reset_index()
    rows = agg['user_id'].map(u2i).values
    cols = agg['edition_id'].map(i2i).values
    vals = np.log1p(agg['w'].values).astype(np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(nu, ni))


def train_ease(mat, item_ids, min_pop=3, lam=25):
    pop = np.array(mat.sum(axis=0)).flatten()
    idx = np.where(pop >= min_pop)[0]
    eids = [item_ids[i] for i in idx]
    n = len(eids)
    log(f"      EASE: {n} items (mp={min_pop} lam={lam})")
    X = mat[:, idx].toarray().astype(np.float32)
    G = X.T @ X + lam * np.eye(n, dtype=np.float32)
    B = np.linalg.inv(G)
    d = np.diag(B).copy()
    B = B / (-d[None, :])
    np.fill_diagonal(B, 0)
    scores = X @ B
    del X, G, B
    gc.collect()
    return scores, eids, {e: i for i, e in enumerate(eids)}


def train_all_models(observed_pairs, enriched_pairs, observed_inter):
    """Train EASE, ALS, SVD, KNN for one fold."""
    t0 = time.time()
    m = {}

    # Main matrix
    mat, uids, iids, u2i, i2i = build_matrix(observed_pairs)
    m.update(mat=mat, uids=uids, iids=iids, u2i=u2i, i2i=i2i)
    log(f"    Matrix: {mat.shape[0]}x{mat.shape[1]} nnz={mat.nnz}")

    # EASE main
    log("    EASE main...")
    t1 = time.time()
    esc, eeids, ee2i = train_ease(mat, iids, 3, 25)
    m.update(ease_scores=esc, ease_eids=eeids, ease_e2i=ee2i)
    log(f"      {time.time()-t1:.0f}s")

    # Enriched matrix + EASE
    mat_e, uids_e, iids_e, u2i_e, i2i_e = build_matrix(enriched_pairs)
    log(f"    Enriched: {mat_e.shape[0]}x{mat_e.shape[1]} nnz={mat_e.nnz}")
    log("    EASE enriched...")
    t1 = time.time()
    esc_e, eeids_e, ee2i_e = train_ease(mat_e, iids_e, 5, 25)
    m.update(mat_e=mat_e, u2i_e=u2i_e, i2i_e=i2i_e, iids_e=iids_e,
             ease_scores_e=esc_e, ease_eids_e=eeids_e, ease_e2i_e=ee2i_e)
    log(f"      {time.time()-t1:.0f}s")

    # SVD
    log("    SVD...")
    t1 = time.time()
    mat_w = build_weighted_matrix(observed_inter, u2i, i2i, len(uids), len(iids))
    k = min(SVD_K, min(mat_w.shape) - 2)
    if k < 2:
        k = 2
    U, sigma, Vt = svds(mat_w.astype(np.float64), k=k)
    order = np.argsort(-sigma)
    U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]
    sq = np.sqrt(sigma).astype(np.float32)
    m['svd_user'] = (U * sq[None, :]).astype(np.float32)
    m['svd_item'] = (Vt.T * sq[None, :]).astype(np.float32)
    m['svd_k'] = k
    del U, sigma, Vt, mat_w
    log(f"      {time.time()-t1:.0f}s k={k}")

    # ALS
    m['als_uf'] = None
    m['als_if'] = None
    if HAS_IMPLICIT:
        log("    ALS...")
        t1 = time.time()
        try:
            als = AlternatingLeastSquares(
                factors=ALS_FACTORS, iterations=ALS_ITERS,
                regularization=0.01, use_gpu=False, random_state=42)
            als.fit(mat)
            m['als_uf'] = np.array(als.user_factors).astype(np.float32)
            m['als_if'] = np.array(als.item_factors).astype(np.float32)
            m['als_model'] = als
            log(f"      {time.time()-t1:.0f}s")
        except Exception as e:
            log(f"      ALS failed: {e}")

    # KNN similarity matrix
    log("    KNN...")
    t1 = time.time()
    pop = np.array(mat.sum(axis=0)).flatten()
    pidx = np.where(pop >= 3)[0]
    peids = [iids[i] for i in pidx]
    kk = min(50, len(peids) - 1)
    if kk < 1:
        kk = 1
    vecs = mat[:, pidx].T.tocsr()
    nn = NearestNeighbors(n_neighbors=kk, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(vecs)
    dists, idxs_nn = nn.kneighbors(vecs)
    rs, cs, vs = [], [], []
    for i in range(len(peids)):
        fi = i2i[peids[i]]
        for j in range(kk):
            sim = 1.0 - dists[i][j]
            if sim > 0.01:
                rs.append(fi)
                cs.append(i2i[peids[idxs_nn[i][j]]])
                vs.append(float(sim))
    m['knn_sim'] = csr_matrix((vs, (rs, cs)), shape=(len(iids), len(iids)))
    del vecs, dists, idxs_nn, nn, rs, cs, vs
    log(f"      {time.time()-t1:.0f}s nnz={m['knn_sim'].nnz}")

    log(f"    All models: {time.time()-t0:.0f}s")
    return m


# ======================== CANDIDATE GENERATION ========================

def generate_candidates(target_users, m, all_obs, user_bks,
                        eid2author, author_eds, lang119, eid2bid):
    t0 = time.time()
    u2i = m['u2i']; iids = m['iids']; mat = m['mat']
    u2i_e = m['u2i_e']
    esc_e = m['ease_scores_e']; eeids_e = m['ease_eids_e']
    svd_ue = m['svd_user']; svd_ie = m['svd_item']
    als_uf = m['als_uf']; als_if = m['als_if']
    knn_sim = m['knn_sim']

    rows = []
    for iu, uid in enumerate(target_users):
        if (iu + 1) % 1000 == 0:
            log(f"      cand {iu+1}/{len(target_users)}")
        uobs = all_obs.get(uid, set())
        ubids = user_bks.get(uid, set())
        cands = defaultdict(int)

        # 1. EASE enriched
        if uid in u2i_e:
            sc = esc_e[u2i_e[uid]]
            ne = len(sc)
            nt = min(N_CAND_EASE, ne)
            if nt > 0:
                tj = np.argpartition(-sc, min(nt - 1, ne - 1))[:nt] if nt < ne else np.arange(ne)
                for j in tj:
                    eid = eeids_e[j]
                    if eid not in uobs:
                        cands[eid] += 1

        # 2. ALS
        if als_uf is not None and uid in u2i:
            uidx = u2i[uid]
            asc = als_uf[uidx] @ als_if.T
            asc[mat[uidx].indices] = -1e18
            ni = len(asc)
            nt = min(N_CAND_ALS, ni)
            if nt > 0:
                tj = np.argpartition(-asc, min(nt - 1, ni - 1))[:nt] if nt < ni else np.arange(ni)
                for j in tj:
                    if asc[j] > -1e17:
                        cands[iids[j]] += 1

        # 3. SVD
        if uid in u2i:
            uidx = u2i[uid]
            ssc = svd_ue[uidx] @ svd_ie.T
            ssc[mat[uidx].indices] = -1e18
            ni = len(ssc)
            nt = min(N_CAND_SVD, ni)
            if nt > 0:
                tj = np.argpartition(-ssc, min(nt - 1, ni - 1))[:nt] if nt < ni else np.arange(ni)
                for j in tj:
                    if ssc[j] > -1e17:
                        cands[iids[j]] += 1

        # 4. KNN
        if uid in u2i:
            uidx = u2i[uid]
            ksc = (mat[uidx] @ knn_sim).toarray().flatten()
            ksc[mat[uidx].indices] = 0
            nnz = int(np.count_nonzero(ksc))
            nt = min(N_CAND_KNN, nnz)
            if nt > 0:
                tj = np.argpartition(-ksc, min(nt - 1, len(ksc) - 1))[:nt] if nt < len(ksc) else np.arange(len(ksc))
                for j in tj:
                    if ksc[j] > 0:
                        cands[iids[j]] += 1

        # 5. Author continuation
        acnts = defaultdict(int)
        for eid in uobs:
            a = eid2author.get(eid)
            if a is not None:
                acnts[a] += 1
        for a, cnt in acnts.items():
            if cnt >= 2 and a in author_eds:
                for eid in author_eds[a]:
                    if eid not in uobs:
                        cands[eid] += 1

        # Filter
        for eid, ns in cands.items():
            if eid in uobs:
                continue
            if eid not in lang119:
                continue
            bid = eid2bid.get(eid)
            if bid is not None and bid in ubids:
                continue
            rows.append((uid, eid, ns))

    df = pd.DataFrame(rows, columns=['user_id', 'edition_id', 'n_sources'])
    log(f"    Candidates: {len(df)} ({len(df)/max(len(target_users),1):.0f}/user) {time.time()-t0:.0f}s")
    return df


# ======================== FEATURE ENGINEERING ========================

def compute_features(cdf, m, obs_enr, obs_inter, ed_info,
                     eid2bid, eid2author, eid2pub, bg_map,
                     users_df, eid2emb, tgt_set, echo_inter=None):
    t0 = time.time()
    u2i = m['u2i']; i2i = m['i2i']; iids = m['iids']; mat = m['mat']
    svd_k = m['svd_k']
    uid_a = cdf['user_id'].values
    eid_a = cdf['edition_id'].values
    n = len(cdf)

    # Index arrays
    uidx = np.array([u2i.get(u, -1) for u in uid_a], dtype=np.int64)
    iidx = np.array([i2i.get(e, -1) for e in eid_a], dtype=np.int64)
    valid = (uidx >= 0) & (iidx >= 0)

    # --- EASE ---
    log("      feat: EASE")
    ee2i = m['ease_e2i']
    eidx_ease = np.array([ee2i.get(e, -1) for e in eid_a], dtype=np.int64)
    ve = (uidx >= 0) & (eidx_ease >= 0)
    fe = np.zeros(n, dtype=np.float32)
    if ve.any():
        fe[ve] = m['ease_scores'][uidx[ve], eidx_ease[ve]]
    cdf['f_ease'] = fe

    # --- SVD ---
    log("      feat: SVD")
    su = np.zeros((n, svd_k), dtype=np.float32)
    si = np.zeros((n, svd_k), dtype=np.float32)
    if valid.any():
        su[valid] = m['svd_user'][uidx[valid]]
        si[valid] = m['svd_item'][iidx[valid]]
    dot_s = np.sum(su * si, axis=1)
    un_s = np.linalg.norm(su, axis=1)
    in_s = np.linalg.norm(si, axis=1)
    cos_s = dot_s / (un_s * in_s + 1e-9)
    cos_s = np.clip(cos_s, -1.0, 1.0)
    cdf['f_svd'] = dot_s
    cdf['f_svd_cos'] = cos_s
    cdf['f_svd_angle'] = np.arccos(cos_s)
    cdf['f_svd_un'] = un_s
    cdf['f_svd_in'] = in_s
    cdf['f_svd_euc'] = np.linalg.norm(su - si, axis=1)
    proj_s = dot_s / (un_s + 1e-9)
    cdf['f_svd_proj'] = proj_s
    cdf['f_svd_orth'] = np.sqrt(np.maximum(0.0, in_s**2 - proj_s**2))
    ew = su * si
    for d in range(min(10, svd_k)):
        cdf[f'f_svd_ew{d}'] = ew[:, d]
    del su, si, ew

    # --- ALS ---
    if m['als_uf'] is not None:
        log("      feat: ALS")
        au = np.zeros((n, ALS_FACTORS), dtype=np.float32)
        ai = np.zeros((n, ALS_FACTORS), dtype=np.float32)
        if valid.any():
            au[valid] = m['als_uf'][uidx[valid]]
            ai[valid] = m['als_if'][iidx[valid]]
        dot_a = np.sum(au * ai, axis=1)
        un_a = np.linalg.norm(au, axis=1)
        in_a = np.linalg.norm(ai, axis=1)
        cdf['f_als'] = dot_a
        cdf['f_als_cos'] = dot_a / (un_a * in_a + 1e-9)
        cdf['f_als_un'] = un_a
        cdf['f_als_in'] = in_a
        del au, ai
    else:
        for c in ['f_als', 'f_als_cos', 'f_als_un', 'f_als_in']:
            cdf[c] = 0.0

    # --- KNN ---
    log("      feat: KNN")
    fk = np.zeros(n, dtype=np.float32)
    grp_idx = cdf.groupby('user_id').indices
    for uid, ridx in grp_idx.items():
        if uid not in u2i:
            continue
        ux = u2i[uid]
        ks = (mat[ux] @ m['knn_sim']).toarray().flatten()
        for ri in ridx:
            ii = i2i.get(eid_a[ri], -1)
            if ii >= 0:
                fk[ri] = ks[ii]
    cdf['f_knn'] = fk

    # --- Item popularity ---
    log("      feat: item stats")
    ipop = np.array(mat.sum(axis=0)).flatten()
    ipop_d = {iids[j]: float(ipop[j]) for j in range(len(iids))}
    cdf['f_pop'] = cdf['edition_id'].map(ipop_d).fillna(0).astype(np.float32)
    cdf['f_pop_log'] = np.log1p(cdf['f_pop'])

    # Item metadata
    epy = ed_info['publication_year'].to_dict()
    ear = ed_info['age_restriction'].to_dict()
    cdf['f_pub_year'] = cdf['edition_id'].map(epy).fillna(0).astype(np.float32)
    cdf['f_age_restr'] = cdf['edition_id'].map(ear).fillna(16).astype(np.float32)

    # Item read ratio
    ie = obs_inter.groupby('edition_id')['event_type'].agg(
        i_read=lambda x: (x == 2).sum(),
        i_events='count'
    )
    ie['ratio'] = (ie['i_read'] / (ie['i_events'] + 1e-9)).astype(np.float32)
    cdf['f_item_read_ratio'] = cdf['edition_id'].map(ie['ratio'].to_dict()).fillna(0).astype(np.float32)

    # Item rating
    ir = obs_inter.dropna(subset=['rating']).groupby('edition_id')['rating']
    ir_mean = ir.mean().to_dict()
    ir_cnt = ir.count().to_dict()
    cdf['f_irat_mean'] = cdf['edition_id'].map(ir_mean).fillna(0).astype(np.float32)
    cdf['f_irat_cnt'] = cdf['edition_id'].map(ir_cnt).fillna(0).astype(np.float32)

    # --- User stats ---
    log("      feat: user stats")
    us = obs_inter.groupby('user_id').agg(
        u_cnt=('edition_id', 'count'),
        u_uniq=('edition_id', 'nunique'),
        u_read=('event_type', lambda x: int((x == 2).sum())),
        u_wish=('event_type', lambda x: int((x == 1).sum())),
    ).reset_index()
    us['u_conv'] = (us['u_read'] / (us['u_cnt'] + 1e-9)).astype(np.float32)
    us['u_wlr'] = (us['u_wish'] / (us['u_cnt'] + 1e-9)).astype(np.float32)

    ur = obs_inter.dropna(subset=['rating']).groupby('user_id')['rating'].mean().reset_index()
    ur.columns = ['user_id', 'u_rat_mean']
    us = us.merge(ur, on='user_id', how='left')
    us['u_rat_mean'] = us['u_rat_mean'].fillna(0).astype(np.float32)

    us = us.merge(users_df[['user_id', 'gender', 'age']], on='user_id', how='left')
    us['gender'] = us['gender'].fillna(0).astype(np.float32)
    us['age'] = us['age'].fillna(0).astype(np.float32)

    cdf = cdf.merge(us, on='user_id', how='left')

    # Rating diff
    cdf['f_rat_diff'] = np.abs(
        cdf['u_rat_mean'].fillna(0) - cdf['f_irat_mean'].fillna(0)
    ).astype(np.float32)

    # User avg pop
    uavgpop = {}
    for uid in tgt_set:
        pops = [ipop_d.get(e, 0) for e in obs_enr.get(uid, set())]
        uavgpop[uid] = float(np.mean(pops)) if pops else 0.0
    cdf['f_uavg_pop'] = cdf['user_id'].map(uavgpop).fillna(0).astype(np.float32)
    cdf['f_pop_vs_u'] = cdf['f_pop'] / (cdf['f_uavg_pop'] + 1e-9)

    # --- Author ---
    log("      feat: author")
    uac = {}
    for uid in tgt_set:
        c = defaultdict(int)
        for eid in obs_enr.get(uid, set()):
            a = eid2author.get(eid)
            if a is not None:
                c[a] += 1
        uac[uid] = c

    ia = np.array([eid2author.get(e, -1) for e in eid_a])
    fah = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        a = ia[idx]
        if a != -1:
            fah[idx] = uac.get(uid_a[idx], {}).get(a, 0)
    cdf['f_auth_hist'] = fah
    cdf['f_same_auth'] = (fah > 0).astype(np.float32)
    cdf['f_heavy_auth'] = (fah >= 3).astype(np.float32)

    # --- Publisher ---
    upc = {}
    for uid in tgt_set:
        c = defaultdict(int)
        for eid in obs_enr.get(uid, set()):
            p = eid2pub.get(eid)
            if p is not None:
                c[p] += 1
        upc[uid] = c

    ip = np.array([eid2pub.get(e, -1) for e in eid_a])
    fph = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        p = ip[idx]
        if p != -1:
            fph[idx] = upc.get(uid_a[idx], {}).get(p, 0)
    cdf['f_pub_aff'] = fph
    cdf['f_same_pub'] = (fph > 0).astype(np.float32)

    # --- Genre ---
    log("      feat: genre")
    ugc = {}
    ugt = {}
    for uid in tgt_set:
        c = defaultdict(int)
        for eid in obs_enr.get(uid, set()):
            bid = eid2bid.get(eid)
            if bid and bid in bg_map:
                for g in bg_map[bid]:
                    c[g] += 1
        ugc[uid] = c
        ugt[uid] = sum(c.values()) if c else 0

    fg = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        uid = uid_a[idx]
        bid = eid2bid.get(eid_a[idx])
        if not bid or bid not in bg_map:
            continue
        uc = ugc.get(uid)
        if not uc:
            continue
        tot = ugt.get(uid, 0)
        if tot == 0:
            continue
        fg[idx] = sum(uc.get(g, 0) for g in bg_map[bid]) / (tot + 1e-9)
    cdf['f_genre'] = fg

    # --- SBERT ---
    if eid2emb:
        log("      feat: SBERT")
        uprof = {}
        for uid in tgt_set:
            embs = [eid2emb[e] for e in obs_enr.get(uid, set()) if e in eid2emb]
            if embs:
                p = np.mean(embs, axis=0)
                nm = np.linalg.norm(p)
                if nm > 0:
                    p = p / nm
                uprof[uid] = p
        fs = np.zeros(n, dtype=np.float32)
        for idx in range(n):
            up = uprof.get(uid_a[idx])
            ip_e = eid2emb.get(eid_a[idx])
            if up is not None and ip_e is not None:
                fs[idx] = float(np.dot(up, ip_e))
        cdf['f_sbert'] = fs
    else:
        cdf['f_sbert'] = 0.0

    # --- Echo features (next-month signals) ---
    if echo_inter is not None and len(echo_inter) > 0:
        log("      feat: echo (next-month)")
        echo_pairs = echo_inter[echo_inter['user_id'].isin(tgt_set)]
        # User-edition set for echo month
        echo_ue = set(zip(echo_pairs['user_id'], echo_pairs['edition_id']))
        # User-author counts in echo month
        echo_ua = defaultdict(lambda: defaultdict(int))
        for u, e in echo_ue:
            a = eid2author.get(e)
            if a is not None:
                echo_ua[u][a] += 1
        # User-genre profile in echo month
        echo_ug = defaultdict(lambda: defaultdict(int))
        for u, e in echo_ue:
            bid = eid2bid.get(e)
            if bid and bid in bg_map:
                for g in bg_map[bid]:
                    echo_ug[u][g] += 1
        # User-edition mapping for echo month
        echo_user_eds = defaultdict(set)
        for u, e in echo_ue:
            echo_user_eds[u].add(e)
        # User SBERT profile in echo month
        echo_uprof = {}
        if eid2emb:
            for uid in tgt_set:
                embs = [eid2emb[e] for e in echo_user_eds.get(uid, set()) if e in eid2emb]
                if embs:
                    p = np.mean(embs, axis=0)
                    nm = np.linalg.norm(p)
                    if nm > 0:
                        p = p / nm
                    echo_uprof[uid] = p

        # Compute per-row
        f_echo_same = np.zeros(n, dtype=np.float32)
        f_echo_auth = np.zeros(n, dtype=np.float32)
        f_echo_genre = np.zeros(n, dtype=np.float32)
        f_echo_sbert = np.zeros(n, dtype=np.float32)
        f_echo_cnt = np.zeros(n, dtype=np.float32)

        for idx in range(n):
            uid = uid_a[idx]
            eid = eid_a[idx]
            # Same book in echo month
            if (uid, eid) in echo_ue:
                f_echo_same[idx] = 1.0
            # Author overlap
            a = eid2author.get(eid)
            if a is not None:
                f_echo_auth[idx] = echo_ua[uid].get(a, 0)
            # Genre overlap
            bid = eid2bid.get(eid)
            if bid and bid in bg_map and uid in echo_ug:
                ug = echo_ug[uid]
                tot = sum(ug.values())
                if tot > 0:
                    f_echo_genre[idx] = sum(ug.get(g, 0) for g in bg_map[bid]) / (tot + 1e-9)
            # Echo user count
            f_echo_cnt[idx] = len(echo_ua.get(uid, {}))
            # SBERT cosine
            if eid2emb:
                up = echo_uprof.get(uid)
                ie = eid2emb.get(eid)
                if up is not None and ie is not None:
                    f_echo_sbert[idx] = float(np.dot(up, ie))

        cdf['f_echo_same'] = f_echo_same
        cdf['f_echo_auth'] = f_echo_auth
        cdf['f_echo_genre'] = f_echo_genre
        cdf['f_echo_sbert'] = f_echo_sbert
        cdf['f_echo_cnt'] = f_echo_cnt
    else:
        cdf['f_echo_same'] = 0.0
        cdf['f_echo_auth'] = 0.0
        cdf['f_echo_genre'] = 0.0
        cdf['f_echo_sbert'] = 0.0
        cdf['f_echo_cnt'] = 0.0

    # --- Z-scores & rank percentiles ---
    log("      feat: z-scores")
    zcols = ['f_ease', 'f_svd', 'f_als', 'f_knn', 'f_pop', 'f_sbert']
    for col in zcols:
        if col in cdf.columns:
            grp = cdf.groupby('user_id')[col]
            gstd = grp.transform('std').fillna(1)
            cdf[f'{col}_z'] = ((cdf[col] - grp.transform('mean')) / (gstd + 1e-9)).astype(np.float32)
            cdf[f'{col}_rp'] = grp.rank(ascending=False, pct=True).astype(np.float32)

    # Fill NaN
    cdf = cdf.fillna(0)
    for col in cdf.columns:
        if cdf[col].dtype == 'object':
            cdf[col] = 0.0

    nf = len([c for c in cdf.columns
              if c.startswith('f_') or c in
              ['n_sources','u_cnt','u_uniq','u_read','u_wish',
               'u_conv','u_wlr','u_rat_mean','gender','age']])
    log(f"      {nf} features, {time.time()-t0:.0f}s")
    return cdf


# ======================== MAIN PIPELINE ========================

log("=" * 60)
log("LTR V3 PIPELINE")
log("=" * 60)
t_total = time.time()

# --- Load data ---
log("\nLoading data...")
fin_inter = pd.read_csv(f'{DATA_DIR}/interactions.csv')
fin_inter['event_ts'] = pd.to_datetime(fin_inter['event_ts'])
editions = pd.read_csv(f'{DATA_DIR}/editions.csv')
targets = pd.read_csv(f'{DATA_DIR}/targets.csv')
users = pd.read_csv(f'{DATA_DIR}/users.csv')
book_genres = pd.read_csv(f'{DATA_DIR}/book_genres.csv')

target_users = targets['user_id'].tolist()
target_set = set(target_users)

ed_info = editions[['edition_id', 'book_id', 'author_id', 'language_id',
                     'publication_year', 'age_restriction', 'publisher_id']].set_index('edition_id')
eid2bid = ed_info['book_id'].to_dict()
eid2author = ed_info['author_id'].to_dict()
eid2pub = ed_info['publisher_id'].to_dict()
bg_map = book_genres.groupby('book_id')['genre_id'].apply(set).to_dict()
author_eds = editions.groupby('author_id')['edition_id'].apply(set).to_dict()
lang119 = set(editions[editions['language_id'] == 119]['edition_id'])

# Enriched data
if os.path.exists(f'{DATA_ENRICHED_DIR}/interactions.csv'):
    enriched_inter = pd.read_csv(f'{DATA_ENRICHED_DIR}/interactions.csv')
else:
    enriched_inter = fin_inter.copy()

fin_pairs = fin_inter[['user_id', 'edition_id']].drop_duplicates()
enr_pairs = enriched_inter[['user_id', 'edition_id']].drop_duplicates()
hack_only = enr_pairs.merge(fin_pairs, on=['user_id', 'edition_id'],
                            how='left', indicator=True)
hack_only = hack_only[hack_only['_merge'] == 'left_only'][['user_id', 'edition_id']]
log(f"Hack-only pairs: {len(hack_only)}")

log(f"Data loaded: {len(fin_inter)} interactions, {len(target_users)} targets, "
    f"{len(editions)} editions")

# --- SBERT ---
eid2emb = {}
if os.path.exists(SBERT_CACHE):
    log(f"Loading SBERT cache: {SBERT_CACHE}")
    eid2emb = pickle.load(open(SBERT_CACHE, 'rb'))
    log(f"  {len(eid2emb)} embeddings, dim={list(eid2emb.values())[0].shape}")
else:
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
        sbert_model = 'cointegrated/rubert-tiny2'
        log(f"Computing SBERT ({sbert_model})...")
        sbert = SentenceTransformer(sbert_model)
        texts = (editions['title'].fillna('') + ' ' + editions['description'].fillna('')).tolist()
        eids_list = editions['edition_id'].tolist()
        raw = sbert.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        log(f"  Raw: {raw.shape}")
        pca = PCA(n_components=32)
        embs = pca.fit_transform(raw).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-9)
        eid2emb = {eid: embs[i] for i, eid in enumerate(eids_list)}
        os.makedirs(os.path.dirname(SBERT_CACHE), exist_ok=True)
        pickle.dump(eid2emb, open(SBERT_CACHE, 'wb'))
        log(f"  Saved cache: {len(eid2emb)} embeddings")
        del sbert, raw, embs
        gc.collect()
    except Exception as e:
        log(f"  SBERT unavailable: {e}")
        log("  Continuing without text features.")

# ======================== TRAINING FOLDS ========================

log("\n" + "=" * 60)
log("TRAINING FOLDS (Jul/Aug/Sep + Oct validation)")
log("=" * 60)

all_months = [
    ('July',      pd.Timestamp('2025-07-01'), pd.Timestamp('2025-08-01'), pd.Timestamp('2025-09-01')),
    ('August',    pd.Timestamp('2025-08-01'), pd.Timestamp('2025-09-01'), pd.Timestamp('2025-10-01')),
    ('September', pd.Timestamp('2025-09-01'), pd.Timestamp('2025-10-01'), pd.Timestamp('2025-11-01')),
    ('October',   pd.Timestamp('2025-10-01'), pd.Timestamp('2025-11-01'), pd.Timestamp('2025-12-01')),
]

all_fold_dfs = []
all_gt = {}

FOLD_CACHE = 'cache_v2/folds_v3_2.pkl'  # v3_2: fixed neg sampling (Arrow bug)
if os.path.exists(FOLD_CACHE):
    log(f"Loading cached folds from {FOLD_CACHE}")
    all_fold_dfs, all_gt = pickle.load(open(FOLD_CACHE, 'rb'))
    log(f"  {len(all_fold_dfs)} folds, {sum(len(d) for d in all_fold_dfs)} rows")
    all_months = []

for month_name, m_start, m_end, echo_end in all_months:
    log(f"\n--- Fold: {month_name} ---")
    tm = time.time()

    # Month target pairs
    mi = fin_inter[(fin_inter['event_ts'] >= m_start) & (fin_inter['event_ts'] < m_end)]
    mt = mi[mi['user_id'].isin(target_set)][['user_id', 'edition_id']].drop_duplicates()

    if len(mt) < 50:
        log(f"  Skip: only {len(mt)} target pairs")
        continue

    # Remove 20%
    np.random.seed(42)
    n_rm = int(len(mt) * 0.20)
    rm_idx = np.random.choice(len(mt), n_rm, replace=False)
    removed = mt.iloc[rm_idx]

    # Ground truth
    gt = defaultdict(set)
    for u, e in zip(removed['user_id'], removed['edition_id']):
        gt[u].add(e)

    # Graded labels
    me_dict = mi.groupby(['user_id', 'edition_id'])['event_type'].max().to_dict()
    rm_labels = {}
    for u, e in zip(removed['user_id'], removed['edition_id']):
        et = me_dict.get((u, e), 1)
        rm_labels[(u, e)] = 2 if et == 2 else 1

    n_read = sum(1 for v in rm_labels.values() if v == 2)
    n_wish = sum(1 for v in rm_labels.values() if v == 1)
    log(f"  Removed {len(removed)} pairs ({len(gt)} users), read={n_read} wish={n_wish}")

    # Build observed (all minus removed)
    rm_df = removed[['user_id', 'edition_id']].copy()
    rm_df['_rm'] = 1
    merged = fin_inter.merge(rm_df, on=['user_id', 'edition_id'], how='left')
    obs_inter = merged[merged['_rm'].isna()].drop(columns=['_rm'])
    obs_pairs = obs_inter[['user_id', 'edition_id']].drop_duplicates()
    enr_obs_pairs = pd.concat([obs_pairs, hack_only]).drop_duplicates()

    # Observed dicts
    all_obs = defaultdict(set)
    for u, e in zip(obs_pairs['user_id'], obs_pairs['edition_id']):
        all_obs[u].add(e)
    obs_enr = defaultdict(set)
    for u, e in zip(enr_obs_pairs['user_id'], enr_obs_pairs['edition_id']):
        obs_enr[u].add(e)
    user_bks = defaultdict(set)
    for uid in target_set:
        for eid in all_obs.get(uid, set()):
            b = eid2bid.get(eid)
            if b is not None:
                user_bks[uid].add(b)

    # Echo interactions (next month after incident)
    echo_inter = fin_inter[(fin_inter['event_ts'] >= m_end) & (fin_inter['event_ts'] < echo_end)]
    log(f"  Echo month: {m_end.strftime('%b')} -> {echo_end.strftime('%b')}, {len(echo_inter)} interactions")

    # Train models
    log("  Training models...")
    models = train_all_models(obs_pairs, enr_obs_pairs, obs_inter)

    # Generate candidates
    log("  Generating candidates...")
    cdf = generate_candidates(
        target_users, models, all_obs, user_bks,
        eid2author, author_eds, lang119, eid2bid)

    if len(cdf) == 0:
        log(f"  No candidates for {month_name}")
        del models; gc.collect()
        continue

    # Compute features
    log("  Computing features...")
    cdf = compute_features(
        cdf, models, obs_enr, obs_inter, ed_info,
        eid2bid, eid2author, eid2pub, bg_map,
        users, eid2emb, target_set, echo_inter=echo_inter)

    # Add labels
    ldf = pd.DataFrame([
        {'user_id': u, 'edition_id': e, 'label': lb}
        for (u, e), lb in rm_labels.items()
    ])
    cdf = cdf.merge(ldf, on=['user_id', 'edition_id'], how='left')
    cdf['label'] = cdf['label'].fillna(0).astype(int)

    # Candidate recall
    pos = int(cdf['label'].sum() > 0)
    pos_n = int((cdf['label'] > 0).sum())
    recall = pos_n / max(len(rm_labels), 1)
    log(f"  Positives in candidates: {pos_n}/{len(rm_labels)} (recall={recall:.1%})")

    # Neg downsampling — avoid groupby().apply() which corrupts user_id on Arrow backend
    if NEG_PER_USER > 0:
        pos_df = cdf[cdf['label'] > 0]
        neg_df = cdf[cdf['label'] == 0].reset_index(drop=True)
        # Pure numpy sampling to bypass Arrow pandas bugs
        _neg_uids = np.array(neg_df['user_id'].tolist())
        _rng = np.random.RandomState(42)
        _keep = []
        for _uid in np.unique(_neg_uids):
            _mask = np.where(_neg_uids == _uid)[0]
            _n = min(len(_mask), NEG_PER_USER)
            _keep.extend(_rng.choice(_mask, _n, replace=False).tolist())
        neg_samp = neg_df.iloc[sorted(_keep)]
        cdf = pd.concat([pos_df, neg_samp], ignore_index=True)
        log(f"  After downsample: {len(cdf)} rows ({(cdf['label']>0).sum()} pos)")

    cdf['month'] = month_name
    cdf['group_id'] = month_name + '_' + cdf['user_id'].astype(str)

    all_fold_dfs.append(cdf)
    all_gt[month_name] = gt

    del models
    gc.collect()
    log(f"  {month_name} done: {time.time()-tm:.0f}s")

if all_fold_dfs and not os.path.exists(FOLD_CACHE):
    os.makedirs('cache_v2', exist_ok=True)
    pickle.dump((all_fold_dfs, all_gt), open(FOLD_CACHE, 'wb'))
    log(f"Saved fold cache: {FOLD_CACHE}")


# ======================== CATBOOST TRAINING ========================

log("\n" + "=" * 60)
log("CATBOOST TRAINING")
log("=" * 60)

train_all = pd.concat(all_fold_dfs, ignore_index=True)

# Sanity check: user_id should never be NaN
_nan_cnt = train_all['user_id'].isna().sum()
if _nan_cnt > 0:
    log(f"  WARNING: {_nan_cnt}/{len(train_all)} rows have NaN user_id — data corruption!")
    train_all = train_all.dropna(subset=['user_id', 'month']).reset_index(drop=True)
    log(f"  After drop: {len(train_all)} rows")

# Integer group_ids — pure Python to bypass ALL Arrow/pandas issues
_months = [str(x) for x in train_all['month'].tolist()]
_uids = [int(x) for x in train_all['user_id'].tolist()]
_month_set = sorted(set(_months))
_month_map = {v: i for i, v in enumerate(_month_set)}
_max_uid = max(_uids) + 1
_keys = np.array([_month_map[_months[i]] * _max_uid + _uids[i]
                   for i in range(len(_months))], dtype=np.int64)
_, _inv = np.unique(_keys, return_inverse=True)
train_all['group_int'] = _inv.astype(np.int32)
# Sort by group for CatBoost contiguity
train_all = train_all.sort_values('group_int').reset_index(drop=True)
del _months, _uids, _keys, _inv

n_groups = train_all['group_int'].nunique()
max_grp = train_all.groupby('group_int').size().max()
log(f"Total training data: {len(train_all)} rows")
log(f"Groups: {n_groups} unique, max_size={max_grp}")
log(f"Per month: {train_all['month'].value_counts().to_dict()}")

# Feature columns
meta_cols = {'user_id', 'edition_id', 'label', 'month', 'group_id', 'group_int'}
feat_cols = sorted([c for c in train_all.columns if c not in meta_cols])
log(f"Features: {len(feat_cols)}")
for c in feat_cols:
    log(f"  {c}")

# Split train / val
train_mask = train_all['month'].isin(['July', 'August', 'September'])
val_mask = train_all['month'] == 'October'

X_tr = train_all.loc[train_mask, feat_cols].values.astype(np.float32)
y_tr = train_all.loc[train_mask, 'label'].values
g_tr = train_all.loc[train_mask, 'group_int'].values

X_va = train_all.loc[val_mask, feat_cols].values.astype(np.float32)
y_va = train_all.loc[val_mask, 'label'].values
g_va = train_all.loc[val_mask, 'group_int'].values

log(f"Train: {len(X_tr)} rows, Val: {len(X_va)} rows")

train_pool = Pool(data=X_tr, label=y_tr, group_id=g_tr)
val_pool = Pool(data=X_va, label=y_va, group_id=g_va)

catboost_task = 'GPU' if USE_GPU else 'CPU'
log(f"CatBoost task_type={catboost_task}")

model = CatBoostRanker(
    loss_function='YetiRank',
    eval_metric='NDCG:top=20',
    depth=8,
    iterations=10000,
    learning_rate=0.03,
    l2_leaf_reg=14.0,
    subsample=0.8,
    bootstrap_type='Bernoulli',
    random_strength=2.0,
    od_type='Iter',
    od_wait=500,
    task_type=catboost_task,
    verbose=500,
    random_seed=42,
)

try:
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
except Exception as e:
    if 'GPU' in str(e) or 'CUDA' in str(e) or 'gpu' in str(e):
        log(f"GPU failed ({e}), falling back to CPU...")
        model = CatBoostRanker(
            loss_function='YetiRank', eval_metric='NDCG:top=20',
            depth=8, iterations=10000, learning_rate=0.03,
            l2_leaf_reg=14.0, subsample=0.8, bootstrap_type='Bernoulli',
            random_strength=2.0, od_type='Iter', od_wait=500,
            task_type='CPU', verbose=500, random_seed=42)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    else:
        raise

best_iter = model.get_best_iteration()
log(f"Best iteration: {best_iter}")

# Feature importance
log("\nFeature importance (top 20):")
imp = model.get_feature_importance(data=train_pool)
for fn, fi in sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:20]:
    log(f"  {fn:<30} {fi:.1f}")

# ======================== OCTOBER VALIDATION ========================

log("\n" + "=" * 60)
log("OCTOBER VALIDATION")
log("=" * 60)

va_df = train_all[val_mask].copy()
va_df['ltr_score'] = model.predict(X_va)

# LTR predictions
ltr_preds = {}
for uid, grp in va_df.groupby('user_id'):
    ltr_preds[uid] = grp.nlargest(20, 'ltr_score')['edition_id'].tolist()
ltr_ndcg = mean_ndcg(ltr_preds, all_gt.get('October', {}))

# RRF baseline on same candidates
rrf_preds = {}
for uid, grp in va_df.groupby('user_id'):
    rankings = []
    for col in ['f_ease', 'f_knn', 'f_auth_hist']:
        if col in grp.columns:
            rankings.append(grp.nlargest(len(grp), col)['edition_id'].tolist())
    scores = defaultdict(float)
    for ranks in rankings:
        for rank, item in enumerate(ranks):
            scores[item] += 1.0 / (60 + rank + 1)
    scored = sorted(scores.items(), key=lambda x: -x[1])
    rrf_preds[uid] = [e for e, _ in scored[:20]]
rrf_ndcg = mean_ndcg(rrf_preds, all_gt.get('October', {}))

log(f"October NDCG@20:")
log(f"  RRF baseline: {rrf_ndcg:.4f}")
log(f"  CatBoost LTR: {ltr_ndcg:.4f}")
log(f"  Delta:        {ltr_ndcg - rrf_ndcg:+.4f} ({(ltr_ndcg-rrf_ndcg)/max(rrf_ndcg,1e-9)*100:+.1f}%)")


# ======================== REAL SUBMISSION ========================

log("\n" + "=" * 60)
log("BUILDING REAL SUBMISSION")
log("=" * 60)
t_sub = time.time()

# Use ALL data
real_pairs = fin_inter[['user_id', 'edition_id']].drop_duplicates()
real_enr_pairs = pd.concat([real_pairs, hack_only]).drop_duplicates()

# Observed dicts
real_obs = defaultdict(set)
for u, e in zip(real_pairs['user_id'], real_pairs['edition_id']):
    real_obs[u].add(e)
real_obs_enr = defaultdict(set)
for u, e in zip(real_enr_pairs['user_id'], real_enr_pairs['edition_id']):
    real_obs_enr[u].add(e)
real_bks = defaultdict(set)
for uid in target_set:
    for eid in real_obs.get(uid, set()):
        b = eid2bid.get(eid)
        if b is not None:
            real_bks[uid].add(b)

# Echo for real submission: November data (the month after the real incident)
real_echo = fin_inter[(fin_inter['event_ts'] >= pd.Timestamp('2025-11-01')) &
                      (fin_inter['event_ts'] < pd.Timestamp('2025-12-01'))]
log(f"Real echo (November): {len(real_echo)} interactions")

# Train models on all data
log("Training real models...")
real_models = train_all_models(real_pairs, real_enr_pairs, fin_inter)

# Generate candidates
log("Generating real candidates...")
real_cdf = generate_candidates(
    target_users, real_models, real_obs, real_bks,
    eid2author, author_eds, lang119, eid2bid)

# Compute features
log("Computing real features...")
real_cdf = compute_features(
    real_cdf, real_models, real_obs_enr, fin_inter, ed_info,
    eid2bid, eid2author, eid2pub, bg_map,
    users, eid2emb, target_set, echo_inter=real_echo)

del real_models
gc.collect()

# Ensure same feature columns
for c in feat_cols:
    if c not in real_cdf.columns:
        real_cdf[c] = 0.0
X_real = real_cdf[feat_cols].values.astype(np.float32)
real_cdf['ltr_score'] = model.predict(X_real)

# Book dedup: keep best edition per book
real_cdf['book_id'] = real_cdf['edition_id'].map(eid2bid).fillna(real_cdf['edition_id'])
real_cdf = real_cdf.sort_values(['user_id', 'ltr_score'], ascending=[True, False])
real_cdf = real_cdf.drop_duplicates(subset=['user_id', 'book_id'], keep='first')

# Popular fallback
cutoff = pd.Timestamp('2025-11-01')
inc_inter = fin_inter[(fin_inter['event_ts'] >= cutoff) & (fin_inter['user_id'].isin(target_set))]
popular = inc_inter.groupby('edition_id').size().sort_values(ascending=False).index.tolist()
popular = [e for e in popular if e in lang119][:200]

# Build submission
results = []
users_done = set()

for uid, grp in real_cdf.groupby('user_id'):
    top20 = grp.head(20)['edition_id'].tolist()

    # Fallback if < 20
    if len(top20) < 20:
        seen = set(top20) | real_obs.get(uid, set())
        bids_seen = real_bks.get(uid, set()) | {eid2bid.get(e) for e in top20 if eid2bid.get(e) is not None}
        for e in popular:
            if e not in seen:
                b = eid2bid.get(e)
                if b is not None and b in bids_seen:
                    continue
                top20.append(e)
                seen.add(e)
                if b is not None:
                    bids_seen.add(b)
                if len(top20) >= 20:
                    break

    for rank, eid in enumerate(top20[:20], 1):
        results.append({'user_id': uid, 'edition_id': eid, 'rank': rank})
    users_done.add(uid)

# Cold users
for uid in target_users:
    if uid not in users_done:
        seen = real_obs.get(uid, set())
        bids_seen = real_bks.get(uid, set())
        rank = 1
        for e in popular:
            if e not in seen:
                b = eid2bid.get(e)
                if b is not None and b in bids_seen:
                    continue
                results.append({'user_id': uid, 'edition_id': e, 'rank': rank})
                seen.add(e)
                if b is not None:
                    bids_seen.add(b)
                rank += 1
                if rank > 20:
                    break
        users_done.add(uid)

sub = pd.DataFrame(results)
log(f"\nSubmission shape: {sub.shape}")
log(f"Unique users: {sub['user_id'].nunique()}")

# Assertions
assert sub['user_id'].nunique() == len(target_users), \
    f"Missing users: {sub['user_id'].nunique()} vs {len(target_users)}"
per_user = sub.groupby('user_id').size()
short = per_user[per_user < 20]
if len(short) > 0:
    log(f"WARNING: {len(short)} users with < 20 predictions!")
    log(f"  min={per_user.min()}, filling with popular...")
    # Fill short users
    for uid in short.index:
        existing = set(sub[sub['user_id'] == uid]['edition_id'])
        cur_rank = len(existing) + 1
        seen = real_obs.get(uid, set()) | existing
        for e in popular:
            if e not in seen:
                results.append({'user_id': uid, 'edition_id': e, 'rank': cur_rank})
                seen.add(e)
                cur_rank += 1
                if cur_rank > 20:
                    break
    sub = pd.DataFrame(results)

# Final checks
assert sub['user_id'].nunique() == len(target_users), "Missing users after fill!"
dup_check = sub.groupby('user_id')['edition_id'].nunique()
assert (dup_check == sub.groupby('user_id').size()).all(), "Duplicate editions!"

sub.to_csv('sub_ltr_v3.csv', index=False)
log(f"\nSaved: sub_ltr_v3.csv")
log(f"Total time: {time.time()-t_total:.0f}s")
log(f"Head:\n{sub.head(25).to_string()}")

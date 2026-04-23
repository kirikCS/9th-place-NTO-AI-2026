"""
Main entry point — runs all pipelines and ensembles via RRF.
Usage: python main.py
Output: submission.csv
"""
import subprocess
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
SETUP_SCRIPTS = [
    "setup_validation.py",  # prepare validation splits
]

PIPELINES = [
    {"script": "pipeline_ltr_v3.py",  "output": "sub_ltr_v3.csv"},
    {"script": "pipeline_v13.py",     "output": "sub_v13.csv"},
    {"script": "pipeline_cf.py",      "output": "sub_cf.csv"},
    {"script": "pipeline_lgb_v4.py",  "output": "sub_lgb_v4.csv"},
]

RRF_K = 60
FINAL_OUTPUT = "submission.csv"

# Data directory — можно переопределить: python main.py --data-dir /path/to/data
DATA_DIR = os.environ.get('DATA_DIR', 'data')
if '--data-dir' in sys.argv:
    DATA_DIR = sys.argv[sys.argv.index('--data-dir') + 1]
os.environ['DATA_DIR'] = DATA_DIR


# ═══════════════════════════════════════════════════════════════
# RUN PIPELINES
# ═══════════════════════════════════════════════════════════════
def run_pipeline(script, expected_output):
    print(f"\n{'='*60}")
    print(f"  Running {script}...")
    print(f"{'='*60}\n")
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"  WARNING: {script} exited with code {result.returncode}")
        return False
    if not os.path.exists(expected_output):
        print(f"  WARNING: {expected_output} not found")
        return False
    df = pd.read_csv(expected_output)
    print(f"  OK: {expected_output} — {df['user_id'].nunique()} users, {len(df)} rows")
    return True


# ═══════════════════════════════════════════════════════════════
# RRF BLEND
# ═══════════════════════════════════════════════════════════════
def load_rankings(path):
    df = pd.read_csv(path)
    rankings = {}
    for uid, grp in df.groupby("user_id"):
        rankings[uid] = grp.sort_values("rank")["edition_id"].tolist()
    return rankings


def rrf_blend(ranking_files, k=RRF_K):
    all_rankings = []
    for f in ranking_files:
        print(f"  Loading {f}...")
        all_rankings.append(load_rankings(f))

    all_users = set()
    for r in all_rankings:
        all_users |= set(r.keys())

    rows = []
    for uid in sorted(all_users):
        scores = defaultdict(float)
        for rankings in all_rankings:
            if uid not in rankings:
                continue
            for rank, eid in enumerate(rankings[uid]):
                scores[eid] += 1.0 / (k + rank + 1)
        top20 = sorted(scores.items(), key=lambda x: -x[1])[:20]
        for rank, (eid, _) in enumerate(top20, 1):
            rows.append({"user_id": uid, "edition_id": int(eid), "rank": rank})

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Run setup scripts first
    for s in SETUP_SCRIPTS:
        print(f"\n  Setup: {s}")
        subprocess.run([sys.executable, s], cwd=os.path.dirname(os.path.abspath(__file__)))

    successful = []

    for p in PIPELINES:
        ok = run_pipeline(p["script"], p["output"])
        if ok:
            successful.append(p["output"])

    if len(successful) == 0:
        print("ERROR: No pipelines produced output!")
        sys.exit(1)

    if len(successful) == 1:
        # Single pipeline — just copy
        print(f"\nOnly 1 pipeline succeeded, using directly.")
        df = pd.read_csv(successful[0])
        df["edition_id"] = df["edition_id"].astype(int)
        df.to_csv(FINAL_OUTPUT, index=False)
    else:
        # Ensemble via RRF
        print(f"\n{'='*60}")
        print(f"  Ensembling {len(successful)} submissions (RRF, K={RRF_K})")
        print(f"{'='*60}")
        df = rrf_blend(successful, k=RRF_K)
        df.to_csv(FINAL_OUTPUT, index=False)

    print(f"\n  DONE: {FINAL_OUTPUT}")
    print(f"  Users: {df['user_id'].nunique()}, Rows: {len(df)}")

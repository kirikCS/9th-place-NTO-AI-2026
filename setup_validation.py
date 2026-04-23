"""
Setup validation splits for CF ensemble pipeline.
Hides 20% of October positives for Optuna weight tuning.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# Load interactions
_data_dir = os.environ.get('DATA_DIR', 'data')
data_path = Path(f"{_data_dir}/interactions.csv")
if not data_path.exists():
    data_path = Path(f"../{_data_dir}/interactions.csv")
df = pd.read_csv(data_path)
df['event_ts'] = pd.to_datetime(df['event_ts'])

print(f"Date range: {df['event_ts'].min()} to {df['event_ts'].max()}")

# Define T as the max date
T = df['event_ts'].max()
print(f"Max date (T): {T}")

# The last 30 days are the "lost items" period in the real task (test set)
# We want to simulate this by taking 30 days BEFORE that.
# Test period (real): [T - 30 days, T]
# Our validation window: [T - 60 days, T - 30 days]

val_start = pd.Timestamp("2025-10-01")
val_end = pd.Timestamp("2025-11-01")

print(f"Validation window: {val_start} to {val_end}")

# Split data
train_df = df[df['event_ts'] < val_start].copy()
val_window_df = df[(df['event_ts'] >= val_start) & (df['event_ts'] < val_end)].copy()
future_df = df[df['event_ts'] >= val_end].copy()

print(f"Train (past) samples: {len(train_df)}")
print(f"Validation window samples: {len(val_window_df)}")
print(f"Future samples: {len(future_df)}")

# Identify positives in the validation window
# Positive = event_type == 1 (wishlist) or 2 (read)
positives = val_window_df[val_window_df['event_type'].isin([1, 2])].copy()
print(f"Total positives in window: {len(positives)}")

# Shuffle and hide 20%
np.random.seed(42)
positives = positives.sample(frac=1).reset_index(drop=True)
split_idx = int(len(positives) * 0.2)

hidden_positives = positives.iloc[:split_idx]
remaining_positives = positives.iloc[split_idx:]

print(f"Hidden positives: {len(hidden_positives)}")
print(f"Remaining positives: {len(remaining_positives)}")

# Create the observable dataset (train + non-positives in window + remaining positives in window + future)
non_positives_in_window = val_window_df[~val_window_df['event_type'].isin([1, 2])]
observable_df = pd.concat([train_df, non_positives_in_window, remaining_positives, future_df]).sort_values('event_ts')

# Save these for later use
output_dir = Path(".")
output_dir.mkdir(exist_ok=True)
observable_df.to_parquet(output_dir / "observable_interactions.parquet")
hidden_positives.to_parquet(output_dir / "hidden_positives.parquet")

print(f"Saved {output_dir / 'observable_interactions.parquet'} and {output_dir / 'hidden_positives.parquet'}")

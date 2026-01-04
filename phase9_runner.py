#!/usr/bin/env python3
"""
phase9_runner.py

This script processes the job application tracker and updates market learning
memory files for Phase 9 – Market Feedback & Adaptation. It computes weighted
aggregates for each resume configuration, cluster, ATS system, and overall
market performance. The results are saved in JSON files named
`config_performance.json`, `cluster_yield.json`, `ats_outcome_patterns.json`,
and `market_performance.json` within the working directory or a specified
output directory.

It assumes that the job application tracker CSV contains the following
columns at minimum:

- application_id (string)
- config_id (string)
- resume_variant (string)
- cover_variant (string)
- cluster (string)
- role_family (string)
- ats_system (string)
- applied_date (date in ISO format, e.g. 2024-09-15)
- current_stage (one of: Auto_Rejected, Rejected_No_Response, Recruiter_Screen,
  Hiring_Manager, Final_Round, Offer)
- failure_mode (one of the enumerated failure modes or blank/unknown)

Optional but recommended columns:
- days_since_apply (integer) – will be computed if missing
- rejection_stage (string) – not used directly here but can exist
- days_to_response (integer) – if present, used for rejection speed metric

The script applies an exponential time-decay weighting (half-life = 60 days)
so that older application outcomes influence metrics less than recent ones.
"""

import os
import json
import math
import datetime
from typing import Dict, Any, Tuple

try:
    import pandas as pd
except ImportError:
    pd = None  # pandas is required; if missing, execution will fail


# Constants for exponential decay (half-life = 60 days)
HALF_LIFE_DAYS = 60.0
LAMBDA = math.log(2.0) / HALF_LIFE_DAYS

# Mapping from current_stage strings to numeric stage scores
STAGE_SCORES = {
    'Auto_Rejected': 0,
    'Rejected_No_Response': 0,
    'Recruiter_Screen': 1,
    'Hiring_Manager': 2,
    'Final_Round': 3,
    'Offer': 4
}

# Enumerated failure modes
FAILURE_MODES = [
    'ELIGIBILITY_GATE',
    'ATS_PACKAGING',
    'DOMAIN_MISMATCH',
    'TOOLING_GAP',
    'SENIORITY_MISMATCH',
    'WEAK_EVIDENCE',
    'EXTERNAL_NOISE',
    'UNKNOWN'
]


def compute_weight(days_since_apply: float) -> float:
    """Return the exponential decay weight given the number of days since apply."""
    # Negative days (future dates) get weight 1.0
    if days_since_apply is None or math.isnan(days_since_apply):
        return 0.0
    if days_since_apply < 0:
        return 1.0
    return math.exp(-LAMBDA * days_since_apply)


def compute_days_since(date_str: str) -> int:
    """Compute the number of days between today and the given ISO date string."""
    try:
        date_obj = datetime.datetime.fromisoformat(str(date_str)).date()
    except Exception:
        return None
    today = datetime.date.today()
    delta = today - date_obj
    return delta.days


def load_tracker(csv_path: str) -> 'pd.DataFrame':
    """Load the job application tracker from a CSV file into a DataFrame."""
    if pd is None:
        raise ImportError("pandas is required to run this script but is not installed.")
    # Read CSV; do not parse dates automatically here
    df = pd.read_csv(csv_path, dtype=str)

    # Normalize column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]

    # Compute days_since_apply if not present or invalid
    if 'days_since_apply' not in df.columns or df['days_since_apply'].isnull().all():
        applied_dates = df.get('applied_date')
        df['days_since_apply'] = applied_dates.apply(compute_days_since)
    else:
        # Try to convert existing column to numeric
        df['days_since_apply'] = pd.to_numeric(df['days_since_apply'], errors='coerce')

    # Compute weights
    df['weight'] = df['days_since_apply'].apply(lambda x: compute_weight(x if x is not None else 0))

    # Compute stage scores
    stage_col = df.get('current_stage', pd.Series(['Auto_Rejected'] * len(df)))
    df['stage_score'] = stage_col.apply(lambda s: STAGE_SCORES.get(str(s).strip(), 0))

    # Handle missing failure_mode values
    if 'failure_mode' not in df.columns:
        df['failure_mode'] = 'UNKNOWN'
    df['failure_mode'] = df['failure_mode'].fillna('UNKNOWN')
    df['failure_mode'] = df['failure_mode'].apply(lambda x: x if x in FAILURE_MODES else 'UNKNOWN')

    # Ensure config_id and cluster exist
    if 'config_id' not in df.columns:
        df['config_id'] = df.get('resume_variant', '') + '_' + df.get('cover_variant', '')
    df['config_id'] = df['config_id'].fillna('unknown')

    if 'cluster' not in df.columns:
        df['cluster'] = 'UNKNOWN'

    # ATS system defaults to 'UNKNOWN' if missing
    if 'ats_system' not in df.columns:
        df['ats_system'] = 'UNKNOWN'
    df['ats_system'] = df['ats_system'].fillna('UNKNOWN')

    return df


def compute_config_metrics(df: 'pd.DataFrame') -> Dict[str, Any]:
    """Compute metrics for each configuration."""
    config_metrics: Dict[str, Any] = {}
    if df.empty:
        return config_metrics

    for config_id, group in df.groupby('config_id'):
        total_weight = group['weight'].sum()
        if total_weight <= 0:
            continue
        recruiter_weight = group[group['stage_score'] >= 1]['weight'].sum()
        recruiter_rate = recruiter_weight / total_weight
        funnel_depth = (group['stage_score'] * group['weight']).sum() / total_weight
        # Average rejection speed: weighted average of days_since_apply for all rows
        avg_rejection_speed = (group['days_since_apply'] * group['weight']).sum() / total_weight
        config_metrics[str(config_id)] = {
            'recruiter_screen_rate': float(recruiter_rate),
            'funnel_depth': float(funnel_depth),
            'average_rejection_speed': float(avg_rejection_speed),
            'num_applications': int(len(group)),
            'is_baseline': False  # will be set later
        }

    # Determine baselines based on system median recruiter_screen_rate
    if config_metrics:
        rates = [m['recruiter_screen_rate'] for m in config_metrics.values()]
        import numpy as np
        median_rate = float(np.median(rates))
        for cfg, metrics in config_metrics.items():
            metrics['is_baseline'] = metrics['recruiter_screen_rate'] >= median_rate

    return config_metrics


def compute_cluster_metrics(df: 'pd.DataFrame') -> Dict[str, Any]:
    """Compute metrics for each role cluster."""
    cluster_metrics: Dict[str, Any] = {}
    if df.empty:
        return cluster_metrics
    for cluster, group in df.groupby('cluster'):
        total_weight = group['weight'].sum()
        if total_weight <= 0:
            continue
        recruiter_weight = group[group['stage_score'] >= 1]['weight'].sum()
        recruiter_rate = recruiter_weight / total_weight
        funnel_depth = (group['stage_score'] * group['weight']).sum() / total_weight
        normalized_depth = funnel_depth / 4.0  # normalizing 0-4 scale to 0-1
        cluster_yield = 0.6 * recruiter_rate + 0.4 * normalized_depth
        cluster_metrics[str(cluster)] = {
            'recruiter_screen_rate': float(recruiter_rate),
            'funnel_depth': float(funnel_depth),
            'cluster_yield': float(cluster_yield),
            'num_applications': int(len(group))
        }
    return cluster_metrics


def compute_ats_patterns(df: 'pd.DataFrame') -> Dict[str, Any]:
    """Compute failure mode frequency distributions for each ATS system."""
    ats_patterns: Dict[str, Any] = {}
    if df.empty:
        return ats_patterns
    for ats, group in df.groupby('ats_system'):
        total_weight = group['weight'].sum()
        fail_freq: Dict[str, float] = {}
        for mode in FAILURE_MODES:
            w_sum = group[group['failure_mode'] == mode]['weight'].sum()
            freq = (w_sum / total_weight) if total_weight > 0 else 0.0
            fail_freq[mode] = float(freq)
        ats_patterns[str(ats)] = fail_freq
    return ats_patterns


def compute_market_metrics(df: 'pd.DataFrame') -> Dict[str, Any]:
    """Compute overall market performance metrics."""
    total_weight = df['weight'].sum()
    if total_weight <= 0:
        return {
            'overall_recruiter_screen_rate': 0.0,
            'overall_funnel_depth': 0.0,
            'overall_cluster_yield': 0.0,
            'last_updated': datetime.date.today().isoformat()
        }
    recruiter_weight = df[df['stage_score'] >= 1]['weight'].sum()
    overall_rate = recruiter_weight / total_weight
    overall_funnel_depth = (df['stage_score'] * df['weight']).sum() / total_weight
    normalized_depth = overall_funnel_depth / 4.0
    overall_cluster_yield = 0.6 * overall_rate + 0.4 * normalized_depth
    return {
        'overall_recruiter_screen_rate': float(overall_rate),
        'overall_funnel_depth': float(overall_funnel_depth),
        'overall_cluster_yield': float(overall_cluster_yield),
        'last_updated': datetime.date.today().isoformat()
    }


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save JSON data to the given path atomically."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    # atomic rename
    os.replace(tmp_path, path)


def run_phase9(tracker_path: str, output_dir: str) -> None:
    """Load data, compute aggregates, and write memory JSON files."""
    df = load_tracker(tracker_path)
    config_metrics = compute_config_metrics(df)
    cluster_metrics = compute_cluster_metrics(df)
    ats_patterns = compute_ats_patterns(df)
    market_metrics = compute_market_metrics(df)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    save_json(config_metrics, os.path.join(output_dir, 'config_performance.json'))
    save_json(cluster_metrics, os.path.join(output_dir, 'cluster_yield.json'))
    save_json(ats_patterns, os.path.join(output_dir, 'ats_outcome_patterns.json'))
    save_json(market_metrics, os.path.join(output_dir, 'market_performance.json'))

    # Optionally print summary stats
    print('Phase 9 completed.')
    print(f"Updated {len(config_metrics)} configuration metrics, {len(cluster_metrics)} cluster metrics, {len(ats_patterns)} ATS patterns.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 9 runner: compute market learning aggregates.')
    parser.add_argument('tracker_csv', help='Path to job_application_tracker.csv')
    parser.add_argument('-o', '--output-dir', default='.', help='Directory to write memory JSON files')
    args = parser.parse_args()
    run_phase9(args.tracker_csv, args.output_dir)


if __name__ == '__main__':
    main()

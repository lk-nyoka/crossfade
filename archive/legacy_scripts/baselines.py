import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Coding Projects\Build or Die\Week 3\Hackathon\Crossfade")
sys.path.insert(0, str(BASE_DIR))

class CountryPairTransitionModel:
    """
    Estimates transition probability P(enters c within 4w | currently in a)
    computed strictly using historical data on or before train_cutoff_date.
    """
    def __init__(self):
        self.transitions: Dict[Tuple[str, str], float] = defaultdict(float)
        self.country_prior: Dict[str, float] = defaultdict(float)
        self.fitted = False

    def fit_from_duckdb(self, con, table_or_query: str, horizon_days: int = 28):
        """High-performance transition matrix calculation directly in DuckDB."""
        res = con.execute(f"""
            WITH first_entries AS (
                SELECT key, cc, MIN(CAST(week_end AS DATE)) as entry_date
                FROM {table_or_query}
                GROUP BY key, cc
            ),
            country_totals AS (
                SELECT cc, COUNT(*) as a_cnt
                FROM first_entries
                GROUP BY cc
            ),
            valid_pairs AS (
                SELECT e1.cc as cc_a, e2.cc as cc_c, COUNT(*) as pair_cnt
                FROM first_entries e1
                JOIN first_entries e2 ON e1.key = e2.key
                WHERE e2.entry_date > e1.entry_date 
                  AND e2.entry_date <= e1.entry_date + INTERVAL {horizon_days} DAYS
                GROUP BY e1.cc, e2.cc
            )
            SELECT p.cc_a, p.cc_c, p.pair_cnt * 1.0 / t.a_cnt as transition_prob
            FROM valid_pairs p
            JOIN country_totals t ON p.cc_a = t.cc
        """).fetchall()
        
        self.transitions = defaultdict(float)
        for a, c, prob in res:
            self.transitions[(a, c)] = float(prob)
            
        priors = con.execute(f"""
            WITH first_entries AS (
                SELECT key, cc, MIN(CAST(week_end AS DATE)) as entry_date
                FROM {table_or_query}
                GROUP BY key, cc
            )
            SELECT cc, COUNT(*) * 1.0 / (SELECT COUNT(DISTINCT key) FROM first_entries) as prior
            FROM first_entries
            GROUP BY cc
        """).fetchall()
        
        self.country_prior = defaultdict(float)
        for c, p in priors:
            self.country_prior[c] = float(p)
            
        self.fitted = True

    def fit(self, historical_df: pd.DataFrame, horizon_days: int = 28):
        """
        historical_df has columns: [key, cc, week_end]
        """
        # Find all country first-entry dates per song
        first_entries = (
            historical_df.groupby(["key", "cc"])["week_end"]
            .min()
            .reset_index()
            .rename(columns={"week_end": "entry_date"})
        )
        
        # Self-join to find pairs (song, country_a, country_c) where c was entered after a within horizon
        merged = pd.merge(first_entries, first_entries, on="key", suffixes=("_a", "_c"))
        # Only where a was entered on or before c, and c entered within horizon_days of a
        diff_days = (merged["entry_date_c"] - merged["entry_date_a"]).dt.days
        valid_transitions = merged[(diff_days > 0) & (diff_days <= horizon_days)]
        
        pair_counts = valid_transitions.groupby(["cc_a", "cc_c"]).size()
        a_counts = first_entries.groupby("cc").size()
        
        for (a, c), cnt in pair_counts.items():
            denom = a_counts.get(a, 1)
            self.transitions[(a, c)] = cnt / max(denom, 1)
            
        total_songs = first_entries["key"].nunique()
        c_counts = valid_transitions.groupby("cc_c").size()
        for c, cnt in c_counts.items():
            self.country_prior[c] = cnt / max(total_songs, 1)
            
        self.fitted = True

    def predict_pair(self, current_markets: List[str], target_c: str, method: str = "max") -> float:
        if not current_markets:
            return self.country_prior.get(target_c, 0.0)
        
        scores = [self.transitions.get((a, target_c), 0.0) for a in current_markets]
        if method == "max":
            return max(scores) if scores else 0.0
        elif method == "noisy_or":
            # 1 - prod(1 - P_i)
            prod_complement = 1.0
            for s in scores:
                prod_complement *= (1.0 - min(s, 0.999))
            return 1.0 - prod_complement
        return 0.0

class CulturalProximityModel:
    """
    Computes cultural and geographic proximity between current markets and target country
    using country metadata (region, sub-region, language) plus historical chart overlap.
    """
    def __init__(self, country_meta_path: Path):
        self.meta_df = pd.read_csv(country_meta_path).set_index("cc")
        self.jaccard_overlap: Dict[Tuple[str, str], float] = defaultdict(float)

    def fit_chart_overlap(self, historical_df: pd.DataFrame):
        # Compute song sets per country
        country_songs = historical_df.groupby("cc")["key"].apply(set).to_dict()
        countries = list(country_songs.keys())
        for i in range(len(countries)):
            c1 = countries[i]
            s1 = country_songs[c1]
            for j in range(i, len(countries)):
                c2 = countries[j]
                s2 = country_songs[c2]
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                sim = intersection / max(union, 1)
                self.jaccard_overlap[(c1, c2)] = sim
                self.jaccard_overlap[(c2, c1)] = sim

    def similarity(self, c_a: str, c_b: str) -> float:
        if c_a == c_b:
            return 1.0
        meta_a = self.meta_df.loc[c_a] if c_a in self.meta_df.index else None
        meta_b = self.meta_df.loc[c_b] if c_b in self.meta_df.index else None
        
        lang_match = 1.0 if (meta_a is not None and meta_b is not None and meta_a["primary_language"] == meta_b["primary_language"]) else 0.0
        region_match = 1.0 if (meta_a is not None and meta_b is not None and meta_a["sub_region"] == meta_b["sub_region"]) else (
            0.5 if (meta_a is not None and meta_b is not None and meta_a["region"] == meta_b["region"]) else 0.0
        )
        overlap = self.jaccard_overlap.get((c_a, c_b), 0.0)
        
        # Weighted composite similarity
        return 0.4 * lang_match + 0.3 * region_match + 0.3 * overlap

    def predict_pair(self, current_markets: List[str], target_c: str) -> float:
        if not current_markets:
            return 0.0
        return max([self.similarity(a, target_c) for a in current_markets])

def compute_ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Computes Recall@5, Recall@10, NDCG@10, MRR for a single (snapshot, song) ranking list.
    """
    total_pos = np.sum(y_true)
    if total_pos == 0:
        return {}
    
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    
    # Recall@K
    rec_5 = np.sum(y_sorted[:5]) / total_pos
    rec_10 = np.sum(y_sorted[:10]) / total_pos
    
    # NDCG@10
    dcg_10 = np.sum(y_sorted[:10] / np.log2(np.arange(2, min(10, len(y_sorted)) + 2)))
    ideal_sorted = np.sort(y_true)[::-1]
    idcg_10 = np.sum(ideal_sorted[:10] / np.log2(np.arange(2, min(10, len(ideal_sorted)) + 2)))
    ndcg_10 = dcg_10 / max(idcg_10, 1e-9)
    
    # MRR (Mean Reciprocal Rank)
    pos_ranks = np.where(y_sorted == 1)[0]
    mrr = (1.0 / (pos_ranks[0] + 1)) if len(pos_ranks) > 0 else 0.0
    
    return {
        "recall_at_5": rec_5,
        "recall_at_10": rec_10,
        "ndcg_at_10": ndcg_10,
        "mrr": mrr
    }

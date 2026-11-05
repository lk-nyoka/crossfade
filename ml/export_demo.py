"""Step 4 - export compact JSON for the static demo (apps/web)."""
import json
import numpy as np
import pandas as pd
from ml.paths import INTERIM, REPORTS, COUNTRY_META, WEB_DATA

LABELS = {
    "tr_max_now": "Strong corridor from a market it's charting in now",
    "tr_noisy_or": "Several of its markets historically feed this one",
    "tr_max_recent": "A market it just entered often leads here",
    "tr_max": "Historical corridor from one of its markets",
    "tr_mean": "Its markets, on average, lead here",
    "artist_prior_songs": "Artist's catalogue size",
    "artist_target_rate": "Artist has broken into this market before",
    "artist_hits_in_target": "Artist's past songs charted here",
    "views_growth_1w": "Views growing week on week",
    "log_views_now": "High view volume",
    "rank_gain_1w": "Climbing the charts",
    "n_same_region": "Already charting in this region",
    "n_same_subregion": "Already charting next door",
    "n_same_lang": "Already charting in a same-language market",
    "mkt_imports_26w": "This market is open to imported hits",
    "mkt_turnover_26w": "This market's chart turns over fast",
    "mkt_prior": "This market receives many breakouts",
    "weeks_since_last_entry": "Recency of its last new market",
    "n_markets_now": "Number of markets it's charting in",
    "age_weeks": "Song age",
    "mean_rank_now": "Average chart position",
    "best_rank_now": "Best chart position",
    "global_rank": "Global chart position",
    "on_global": "On the global chart",
    "debut_markets": "Size of its debut",
    "n_markets_ever": "Markets reached so far",
    "new_markets_4w": "New markets in the last month",
    "markets_gain_1w": "Markets gained this week",
    "is_simul_launch": "Coordinated global launch",
}


def describe(f, row, cn):
    """Evidence-backed reason text; returns None when the feature's value doesn't support the claim."""
    v = float(row[f]) if f in row and row[f] == row[f] else None
    src = cn.get(str(row.get("top_source")), str(row.get("top_source")))
    if f in ("tr_max_now", "tr_max", "tr_max_recent", "tr_noisy_or", "tr_mean"):
        return f"Corridor: songs charting in {src} entered this market within 4 weeks {row['tr_max']*100:.1f}% of the time"
    if f in ("artist_hits_in_target", "artist_target_rate"):
        n = int(row["artist_hits_in_target"])
        return f"Artist has charted here before ({n} song{'s' if n != 1 else ''})" if n > 0 else None
    if f == "artist_prior_songs":
        n = int(row["artist_prior_songs"])
        return f"Established artist ({n} charting songs)" if n >= 3 else None
    if f == "views_growth_1w" and v is not None and v > 0:
        return f"Views up {np.expm1(v)*100:.0f}% week on week"
    if f == "rank_gain_1w" and v is not None and v > 0:
        if int(row["best_rank_now"] + v) == 101:  # no chart position the week before
            return f"New chart entry this week (#{int(row['best_rank_now'])})"
        return f"Best chart position up {int(v)} places this week"
    if f in ("n_same_region", "n_same_subregion") and v and v > 0:
        return f"Already charting in {int(v)} market{'s' if v != 1 else ''} in the same {'region' if f == 'n_same_region' else 'neighbourhood'}"
    if f == "n_same_lang" and v and v > 0:
        return f"Already in {int(v)} same-language market{'s' if v != 1 else ''}"
    if f == "mkt_imports_26w" and v is not None:
        return f"Open market: ~{v:.1f} imported hits enter its chart each week"
    if f == "n_markets_now" and v:
        return f"Charting in {int(v)} markets right now"
    if f == "log_views_now" and v:
        return f"{np.expm1(v)/1e6:.1f}M weekly views in its biggest market"
    if f == "weeks_since_last_entry" and v is not None and v <= 2 and row["n_markets_ever"] > 1:
        return "Reached a new market in the last 2 weeks"
    return None


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


CN = {}


def main():
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(COUNTRY_META)
    global CN
    CN = dict(zip(meta.cc, meta.country_name))
    countries = {row.cc: {"name": row.country_name, "region": row.region, "sub": row.sub_region}
                 for row in meta.itertuples() if row.cc != "global"}
    songs = pd.read_parquet(INTERIM / "songs.parquet").set_index("song_id")
    charts = pd.read_parquet(INTERIM / "charts.parquet")
    charts = charts[charts.cc != "global"].copy()
    charts["week_end"] = pd.to_datetime(charts["week_end"])
    songs["debut_week"] = pd.to_datetime(songs["debut_week"])
    entries = charts.groupby(["song_id", "cc"]).week_end.min().reset_index()
    trans = pd.read_parquet(INTERIM / "transitions_latest.parquet")
    P = {(a, c): p for a, c, p in zip(trans.a, trans.c, trans.p)}

    def song_card(sid, t, cand, contrib=None, actual=None):
        s = songs.loc[sid]
        t = pd.Timestamp(t)
        now = charts[(charts.song_id == sid) & (charts.week_end == t)].sort_values("rank")
        ent = entries[(entries.song_id == sid) & (entries.week_end <= t)].sort_values("week_end")
        c = cand.sort_values("p", ascending=False).head(8)
        targets = []
        for i, row in c.iterrows():
            reasons = []
            if contrib is not None:
                cv = contrib.loc[i]
                for f, v in cv.sort_values(ascending=False).items():
                    if v <= 0 or len(reasons) == 3:
                        break
                    txt = describe(f, row, CN)
                    if txt and txt not in reasons:
                        reasons.append(txt)
            src = str(row.top_source)
            targets.append({"cc": str(row.cc), "p": r(row.p), "src": src, "src_p": r(row.tr_max),
                            "artist_hits": int(row.artist_hits_in_target), "reasons": reasons,
                            **({"hit": int(row.y)} if actual else {})})
        out = {"id": sid, "title": s.title, "artists": s.artists, "debut": str(s.debut_week.date()),
               "age": int((t - s.debut_week).days // 7), "simul": bool(s.is_simul_launch), "t": str(t.date()),
               "now": [{"cc": a, "rank": int(b)} for a, b in zip(now.cc, now["rank"])],
               "path": [{"cc": a, "w": str(b.date())} for a, b in zip(ent.cc, ent.week_end)],
               "expected": r(float(cand.p.sum()), 2), "targets": targets}
        if actual:
            out["actual"] = actual
        return out

    # ---- live radar: latest chart week ----
    live = pd.read_parquet(INTERIM / "live_predictions.parquet")
    live["t"] = pd.to_datetime(live["t"])
    lc = pd.read_parquet(INTERIM / "live_contrib.parquet")
    lc.index = live.index
    feat_cols = [c for c in lc.columns if c not in ("song_id", "cc")]
    lc = lc[feat_cols]
    live["song_id"] = live.song_id.astype(str)
    live["cc"] = live.cc.astype(str)
    score = live.groupby("song_id").p.sum().sort_values(ascending=False)  # expected number of new markets
    afr = set(meta[meta.region == "Africa"].cc)
    afr_songs = set(live[live.song_id.isin(charts[(charts.week_end == live.t.max()) & charts.cc.isin(afr)].song_id)].song_id)
    pick = list(score.index[:220]) + [s for s in score.index if s in afr_songs][:80]
    pick = list(dict.fromkeys(pick))
    t_live = live.t.max()
    radar = [song_card(sid, t_live, live[live.song_id == sid], lc) for sid in pick]

    # ---- backtest: held-out 2025-26 weeks, outcomes known ----
    te = pd.read_parquet(INTERIM / "test_predictions.parquet")
    te["song_id"] = te.song_id.astype(str); te["cc"] = te.cc.astype(str)
    te = te.rename(columns={"Crossfade_LGBM": "p"})
    te["t"] = pd.to_datetime(te["t"])
    grp = te.groupby("gid")
    gs = pd.DataFrame({"pos": grp.y.sum(), "songs": grp.song_id.first(), "t": grp.t.first(),
                       "simul": grp.is_simul_launch.first()})
    gs = gs[(gs.pos >= 2) & (gs.simul == 0)]
    debut_af = songs.debut_ccs.map(lambda l: any(c in afr for c in l))
    gs["af"] = gs.songs.map(debut_af)
    rng = np.random.default_rng(3)
    gs = gs.sample(frac=1, random_state=1).drop_duplicates("songs")  # one week per song
    chosen = list(gs[gs.af].head(25).index) + \
             list(gs[~gs.af].sample(frac=1, random_state=1).drop_duplicates("songs").head(35).index)
    backtest = []
    for g in chosen:
        cand = te[te.gid == g]
        sid, t = cand.song_id.iloc[0], cand.t.iloc[0]
        actual = sorted(cand[cand.y == 1].cc.tolist())
        backtest.append(song_card(sid, t, cand, None, actual=actual))

    # ---- corridors: strongest country-to-country flows ----
    tr = trans[trans.p > 0].sort_values("p", ascending=False)
    corridors = [{"a": a, "c": c, "p": r(p)} for a, c, p in zip(tr.a, tr.c, tr.p)]

    bench = pd.read_csv(REPORTS / "benchmark.csv")
    slices = pd.read_csv(REPORTS / "slices_test.csv")
    summary = json.load(open(REPORTS / "run_summary.json"))
    shap = pd.read_csv(REPORTS / "shap_importance.csv", index_col=0)["mean_abs_shap"]
    cal = pd.read_csv(REPORTS / "calibration_test.csv")
    model = {
        "summary": summary,
        "benchmark": bench[bench.split.isin(["fold_2022", "fold_2023", "fold_2024", "test_all", "test_emerging", "test_simultaneous_launch"])]
            .round(4).to_dict("records"),
        "slices": slices[slices.slice.isin(["all", "debuted_in_africa"])].round(4).to_dict("records"),
        "shap": [{"f": f, "label": LABELS.get(f, f), "v": r(v)} for f, v in shap.items()],
        "calibration": cal.round(6).to_dict("records"),
        "decision": pd.read_csv(REPORTS / "decision_value.csv").round(4).astype(object).where(lambda d: d.notna(), None).to_dict("records"),
    }
    data = {"countries": countries, "radar": radar, "backtest": backtest, "corridors": corridors[:400],
            "model": model, "live_week": str(t_live.date())}
    js = json.dumps(data, separators=(",", ":"), default=str, allow_nan=False)
    (WEB_DATA / "crossfade.json").write_text(js)
    print(f"radar {len(radar)} songs, backtest {len(backtest)}, corridors {len(corridors)}; {len(js)/1e6:.2f} MB")


if __name__ == "__main__":
    main()

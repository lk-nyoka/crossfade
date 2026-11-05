"""Step 2 - candidates, 4-week labels and leakage-safe features.

Prediction unit: (snapshot week t, song s, target country c)
  * s is charting in >=1 national chart at t and is <= 26 weeks past its national debut
  * c is a national chart (the 'global' chart is a feature, never a target) that s has never entered up to t
  * label = 1 if s first enters c in (t, t+28 days]

Every feature uses data dated <= t only. Country-to-country transition rates are re-estimated
each quarter from events whose 4-week outcome was fully observed before that quarter began.

Rows kept (to fit in memory, documented in docs/EVALUATION.md):
  * eval set: every row of a random 25% of (song, week) groups before 2025, 100% from 2025 on
  * train set: every positive + 5% of negatives (weighted x20) before 2025
  * live set: the latest chart week (2026-10-15), unlabeled, used by the demo
"""
import duckdb
import yaml
from ml.paths import INTERIM, COUNTRY_META, CONFIG

CFG = yaml.safe_load(open(CONFIG))
H = CFG["horizon_weeks"] * 7
BURN_IN = CFG["burn_in_date"]
LAST_LABELED = CFG["max_snapshot_date"]
LIVE = CFG["live_snapshot_date"]
MAX_AGE = CFG["max_song_age_weeks"] * 7
TEST_START = CFG["splits"]["final_test_start"]
ALPHA = 20.0  # Bayesian smoothing strength for transition rates


def main():
    con = duckdb.connect(config={"memory_limit": "5GB", "threads": 2})
    con.execute(f"SET temp_directory='{(INTERIM / 'duck_tmp').as_posix()}'")
    con.execute(f"CREATE VIEW charts AS SELECT * FROM '{(INTERIM / 'charts.parquet').as_posix()}'")
    con.execute(f"CREATE VIEW songs AS SELECT * FROM '{(INTERIM / 'songs.parquet').as_posix()}'")
    con.execute(f"CREATE TABLE meta AS SELECT * FROM read_csv_auto('{COUNTRY_META.as_posix()}')")
    con.execute("CREATE TABLE nat AS SELECT * FROM charts WHERE cc <> 'global'")
    con.execute("CREATE TABLE glob_chart AS SELECT song_id, week_end, rank AS global_rank FROM charts WHERE cc = 'global'")
    con.execute("CREATE TABLE entries AS SELECT song_id, cc, MIN(week_end) AS e FROM nat GROUP BY ALL")
    con.execute("CREATE TABLE ccs AS SELECT DISTINCT cc FROM nat")

    # ---------- candidates + labels ----------
    print("candidates...", flush=True)
    con.execute(f"""
        CREATE TABLE active AS
        SELECT DISTINCT n.song_id, n.week_end AS t
        FROM nat n JOIN songs s USING (song_id)
        WHERE ((n.week_end BETWEEN DATE '{BURN_IN}' AND DATE '{LAST_LABELED}') OR n.week_end = DATE '{LIVE}')
          AND date_diff('day', s.debut_week, n.week_end) <= {MAX_AGE}
    """)
    con.execute(f"""
        CREATE TABLE cand_all AS
        SELECT a.song_id, a.t, c.cc,
               CASE WHEN a.t = DATE '{LIVE}' THEN NULL
                    ELSE (e.e IS NOT NULL AND e.e <= a.t + INTERVAL {H} DAY)::INT END AS y,
               hash(a.song_id || CAST(a.t AS VARCHAR)) % 100 AS gbucket,
               hash(a.song_id || CAST(a.t AS VARCHAR) || c.cc) % 100 AS rbucket
        FROM active a CROSS JOIN ccs c
        LEFT JOIN entries e ON e.song_id = a.song_id AND e.cc = c.cc
        WHERE e.e IS NULL OR e.e > a.t
    """)
    con.execute(f"""
        CREATE TABLE cand AS
        SELECT song_id, t, cc, y,
               CASE WHEN t = DATE '{LIVE}' THEN 'live'
                    WHEN t >= DATE '{TEST_START}' THEN 'test' ELSE 'dev' END AS period,
               (t >= DATE '{TEST_START}' OR gbucket < 25) AS in_eval,
               (t < DATE '{TEST_START}' AND (y = 1 OR rbucket < 5)) AS in_train,
               CASE WHEN y = 1 THEN 1.0 ELSE 20.0 END AS w
        FROM cand_all
        WHERE t >= DATE '{TEST_START}' OR gbucket < 25 OR y = 1 OR rbucket < 5
    """)
    stats = con.sql(f"""SELECT COUNT(*) n, SUM(y) pos, SUM(y)/COUNT(y) rate,
                         COUNT(DISTINCT (song_id, t)) groups FROM cand_all WHERE y IS NOT NULL""").fetchall()[0]
    print(f"  full candidate universe: {stats[0]:,} rows, {stats[1]:,} positives, base rate {stats[2]:.4%}, {stats[3]:,} groups")
    con.execute("DROP TABLE cand_all")
    print("  kept rows:", con.sql("SELECT period, COUNT(*), SUM(y) FROM cand GROUP BY 1 ORDER BY 1").fetchall(), flush=True)

    # ---------- song state at t (history <= t only) ----------
    print("song features...", flush=True)
    con.execute("CREATE TABLE gt AS SELECT DISTINCT song_id, t FROM cand")
    con.execute("""
        CREATE TABLE song_now AS
        SELECT g.song_id, g.t,
               COUNT(*) AS n_markets_now, MIN(n.rank) AS best_rank_now, AVG(n.rank) AS mean_rank_now,
               LN(1 + MAX(n.views)) AS log_views_now
        FROM gt g JOIN nat n ON n.song_id = g.song_id AND n.week_end = g.t
        GROUP BY ALL
    """)
    con.execute("""
        CREATE TABLE song_prev AS
        SELECT g.song_id, g.t, MIN(n.rank) AS best_rank_prev, LN(1 + MAX(n.views)) AS log_views_prev,
               COUNT(*) AS n_markets_prev
        FROM gt g JOIN nat n ON n.song_id = g.song_id AND n.week_end = g.t - INTERVAL 7 DAY
        GROUP BY ALL
    """)
    con.execute(f"""
        CREATE TABLE song_hist AS
        SELECT g.song_id, g.t,
               COUNT(*) AS n_markets_ever,
               COUNT(*) FILTER (WHERE e.e > g.t - INTERVAL {H} DAY) AS new_markets_4w,
               date_diff('day', MAX(e.e), g.t) / 7 AS weeks_since_last_entry
        FROM gt g JOIN entries e ON e.song_id = g.song_id AND e.e <= g.t
        GROUP BY g.song_id, g.t
    """)
    con.execute("""
        CREATE TABLE song_feat AS
        SELECT g.song_id, g.t,
               date_diff('day', s.debut_week, g.t) / 7 AS age_weeks,
               s.is_simul_launch::INT AS is_simul_launch, s.debut_markets,
               sn.n_markets_now, sn.best_rank_now, sn.mean_rank_now, sn.log_views_now,
               COALESCE(sp.best_rank_prev, 101) - sn.best_rank_now AS rank_gain_1w,
               sn.log_views_now - sp.log_views_prev AS views_growth_1w,
               sn.n_markets_now - COALESCE(sp.n_markets_prev, 0) AS markets_gain_1w,
               sh.n_markets_ever, sh.new_markets_4w, sh.weeks_since_last_entry,
               (gl.global_rank IS NOT NULL)::INT AS on_global, COALESCE(gl.global_rank, 101) AS global_rank,
               s.primary_artist
        FROM gt g
        JOIN songs s USING (song_id)
        JOIN song_now sn USING (song_id, t)
        LEFT JOIN song_prev sp USING (song_id, t)
        JOIN song_hist sh USING (song_id, t)
        LEFT JOIN glob_chart gl ON gl.song_id = g.song_id AND gl.week_end = g.t
    """)

    # ---------- target-market openness (trailing 26 weeks, <= t) ----------
    print("market features...", flush=True)
    con.execute("""
        CREATE TABLE weekly_in AS
        SELECT w.cc, w.week_end,
               COUNT(e.song_id) AS new_entries,
               COUNT(e.song_id) FILTER (WHERE s.debut_week < e.e) AS imported_entries
        FROM (SELECT DISTINCT cc, week_end FROM nat) w
        LEFT JOIN entries e ON e.cc = w.cc AND e.e = w.week_end
        LEFT JOIN songs s ON s.song_id = e.song_id
        GROUP BY ALL
    """)
    con.execute("""
        CREATE TABLE market_feat AS
        SELECT cc, week_end AS t,
               AVG(new_entries) OVER w AS mkt_turnover_26w,
               AVG(imported_entries) OVER w AS mkt_imports_26w
        FROM weekly_in
        WINDOW w AS (PARTITION BY cc ORDER BY week_end ROWS BETWEEN 25 PRECEDING AND CURRENT ROW)
    """)

    # ---------- quarterly, outcome-resolved transition rates P(a -> c) ----------
    print("transition rates...", flush=True)
    con.execute(f"""
        CREATE TABLE trans_events AS
        SELECT ea.cc AS a, c.cc AS c,
               ea.e + INTERVAL {H} DAY AS resolved,
               (ec.e IS NOT NULL AND ec.e <= ea.e + INTERVAL {H} DAY)::INT AS hit
        FROM entries ea CROSS JOIN ccs c
        LEFT JOIN entries ec ON ec.song_id = ea.song_id AND ec.cc = c.cc
        WHERE c.cc <> ea.cc AND (ec.e IS NULL OR ec.e > ea.e)
    """)
    con.execute("""
        CREATE TABLE quarters AS
        SELECT DISTINCT date_trunc('quarter', t)::DATE AS q FROM gt
    """)
    con.execute(f"""
        CREATE TABLE trans_q AS
        WITH agg AS (
            SELECT q.q, te.a, te.c, SUM(te.hit) AS hits, COUNT(*) AS n
            FROM quarters q JOIN trans_events te ON te.resolved < q.q
            GROUP BY q.q, te.a, te.c
        ),
        prior AS (SELECT q, c, SUM(hits) / SUM(n) AS p_c FROM agg GROUP BY ALL),
        glob_prior AS (SELECT q, SUM(hits) / SUM(n) AS p0 FROM agg GROUP BY ALL)
        SELECT q.q, a.cc AS a, c.cc AS c,
               (COALESCE(agg.hits, 0) + {ALPHA} * COALESCE(pr.p_c, gp.p0, 0.002)) /
               (COALESCE(agg.n, 0) + {ALPHA}) AS p,
               COALESCE(pr.p_c, gp.p0, 0.002) AS p_c
        FROM quarters q CROSS JOIN ccs a CROSS JOIN ccs c
        LEFT JOIN agg ON agg.q = q.q AND agg.a = a.cc AND agg.c = c.cc
        LEFT JOIN prior pr ON pr.q = q.q AND pr.c = c.cc
        LEFT JOIN glob_prior gp ON gp.q = q.q
        WHERE a.cc <> c.cc
    """)
    # export the latest matrix for the demo's corridor map
    con.execute(f"""COPY (SELECT * FROM trans_q WHERE q = (SELECT MAX(q) FROM trans_q))
                    TO '{(INTERIM / 'transitions_latest.parquet').as_posix()}' (FORMAT parquet)""")

    # ---------- pair features: song's markets x target ----------
    print("pair features...", flush=True)
    con.execute(f"""
        CREATE TABLE song_mk AS
        SELECT g.song_id, g.t, e.cc AS a, e.e,
               (n.song_id IS NOT NULL) AS charting_now
        FROM gt g JOIN entries e ON e.song_id = g.song_id AND e.e <= g.t
        LEFT JOIN nat n ON n.song_id = g.song_id AND n.cc = e.cc AND n.week_end = g.t
    """)
    con.execute("""
        CREATE TABLE pair_feat AS
        SELECT c.song_id, c.t, c.cc,
               MAX(tq.p) AS tr_max,
               1 - EXP(SUM(LN(1 - LEAST(tq.p, 0.999)))) AS tr_noisy_or,
               AVG(tq.p) AS tr_mean,
               MAX(tq.p) FILTER (WHERE sm.charting_now) AS tr_max_now,
               MAX(tq.p) FILTER (WHERE sm.e > c.t - INTERVAL 28 DAY) AS tr_max_recent,
               ANY_VALUE(tq.p_c) AS mkt_prior,
               SUM((ma.primary_language = mc.primary_language)::INT) AS n_same_lang,
               SUM((ma.sub_region = mc.sub_region)::INT) AS n_same_subregion,
               SUM((ma.region = mc.region)::INT) AS n_same_region,
               ARG_MAX(sm.a, tq.p) AS top_source
        FROM cand c
        JOIN song_mk sm ON sm.song_id = c.song_id AND sm.t = c.t
        JOIN trans_q tq ON tq.q = date_trunc('quarter', c.t)::DATE AND tq.a = sm.a AND tq.c = c.cc
        JOIN meta ma ON ma.cc = sm.a
        JOIN meta mc ON mc.cc = c.cc
        GROUP BY c.song_id, c.t, c.cc
    """)

    # ---------- artist track record in the target market (<= t) ----------
    print("artist features...", flush=True)
    con.execute("""
        CREATE TABLE artist_cc AS
        SELECT s.primary_artist, e.cc, e.e,
               COUNT(*) OVER (PARTITION BY s.primary_artist, e.cc ORDER BY e.e
                              RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS hits
        FROM entries e JOIN songs s USING (song_id)
        WHERE s.primary_artist <> ''
    """)
    con.execute("CREATE TABLE artist_cc_d AS SELECT primary_artist, cc, e, MAX(hits) hits FROM artist_cc GROUP BY ALL")
    con.execute("""
        CREATE TABLE artist_songs AS
        SELECT primary_artist, debut_week,
               COUNT(*) OVER (PARTITION BY primary_artist ORDER BY debut_week
                              RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS n_songs
        FROM songs WHERE primary_artist <> ''
    """)
    con.execute("CREATE TABLE artist_songs_d AS SELECT primary_artist, debut_week, MAX(n_songs) n_songs FROM artist_songs GROUP BY ALL")

    # ---------- assemble ----------
    print("assembling...", flush=True)
    con.execute(f"""
        CREATE TABLE ds AS
        WITH base AS (
            SELECT c.*, sf.* EXCLUDE (song_id, t), pf.* EXCLUDE (song_id, t, cc),
                   mf.mkt_turnover_26w, mf.mkt_imports_26w
            FROM cand c
            JOIN song_feat sf USING (song_id, t)
            JOIN pair_feat pf USING (song_id, t, cc)
            LEFT JOIN market_feat mf USING (cc, t)
        )
        SELECT b.*,
               COALESCE(ac.hits, 0) AS artist_hits_in_target,
               COALESCE(asg.n_songs, 0) AS artist_prior_songs
        FROM base b
        ASOF LEFT JOIN artist_cc_d ac
          ON ac.primary_artist = b.primary_artist AND ac.cc = b.cc AND ac.e <= b.t
        ASOF LEFT JOIN artist_songs_d asg
          ON asg.primary_artist = b.primary_artist AND asg.debut_week <= b.t
    """)
    con.execute("""ALTER TABLE ds ADD COLUMN artist_target_rate DOUBLE;
                   UPDATE ds SET artist_target_rate = artist_hits_in_target / GREATEST(artist_prior_songs, 1);""")
    out = (INTERIM / "dataset.parquet").as_posix()
    con.execute(f"COPY ds TO '{out}' (FORMAT parquet)")
    print(con.sql("SELECT period, COUNT(*) n, SUM(y) pos, SUM(in_eval::INT) eval_rows, SUM(in_train::INT) train_rows FROM ds GROUP BY 1 ORDER BY 1").df())
    lost = con.sql("SELECT (SELECT COUNT(*) FROM cand) - (SELECT COUNT(*) FROM ds)").fetchall()[0][0]
    print("rows lost in joins (should be 0):", lost)


if __name__ == "__main__":
    main()

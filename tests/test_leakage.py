"""Leakage and label-integrity checks on the built dataset (run after `python -m ml.pipelines.build_dataset`).
Each check recomputes a quantity from the raw chart table using only data dated <= snapshot t."""
import duckdb
import pytest
from ml.paths import INTERIM

DS = INTERIM / "dataset.parquet"
CH = INTERIM / "charts.parquet"
pytestmark = pytest.mark.skipif(not DS.exists(), reason="dataset not built")


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect()
    c.execute(f"CREATE VIEW ds AS SELECT * FROM '{DS.as_posix()}'")
    c.execute(f"CREATE VIEW nat AS SELECT * FROM '{CH.as_posix()}' WHERE cc <> 'global'")
    c.execute("CREATE TABLE smp AS SELECT * FROM ds USING SAMPLE 20000 ROWS (reservoir, 42)")
    return c


def one(con, q):
    return con.sql(q).fetchall()[0][0]


def test_global_is_never_a_target(con):
    assert one(con, "SELECT COUNT(*) FROM ds WHERE cc = 'global'") == 0


def test_target_not_entered_before_snapshot(con):
    assert one(con, """SELECT COUNT(*) FROM smp s JOIN nat n
                       ON n.song_id = s.song_id AND n.cc = s.cc AND n.week_end <= s.t""") == 0


def test_label_matches_future_entry_within_4_weeks(con):
    bad = one(con, """
        SELECT COUNT(*) FROM smp s
        LEFT JOIN (SELECT song_id, cc, MIN(week_end) e FROM nat GROUP BY 1, 2) f USING (song_id, cc)
        WHERE s.y IS NOT NULL
          AND s.y <> (f.e IS NOT NULL AND f.e > s.t AND f.e <= s.t + INTERVAL 28 DAY)::INT""")
    assert bad == 0


def test_market_count_uses_only_past(con):
    bad = one(con, """
        SELECT COUNT(*) FROM smp s
        JOIN (SELECT s2.song_id, s2.t, COUNT(DISTINCT n.cc) k FROM (SELECT DISTINCT song_id, t FROM smp) s2
              JOIN nat n ON n.song_id = s2.song_id AND n.week_end <= s2.t GROUP BY 1, 2) r USING (song_id, t)
        WHERE r.k <> s.n_markets_ever""")
    assert bad == 0


def test_song_age_within_window(con):
    assert one(con, "SELECT MAX(age_weeks) FROM ds") <= 26


def test_test_period_rows_never_used_for_training(con):
    assert one(con, "SELECT COUNT(*) FROM ds WHERE in_train AND t >= DATE '2025-01-01'") == 0


def test_live_week_has_no_labels(con):
    assert one(con, "SELECT COUNT(*) FROM ds WHERE period = 'live' AND y IS NOT NULL") == 0

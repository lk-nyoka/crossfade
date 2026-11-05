"""Step 1 - canonical song identities and a clean (song, country, week) chart table.

Outputs (data/interim/):
  key_map.parquet      raw YouTube key -> canonical song_id
  songs.parquet        one row per canonical song (title, artists, primary artist, debut info)
  charts.parquet       one row per (song_id, cc, week_end): best rank, max views. 'global' kept separately.
"""
from collections import Counter
import duckdb
import pandas as pd
from ml.paths import PARQUET_GLOB, INTERIM
from ml.pipelines.song_identity import normalize_song_identity

SIMUL_THRESHOLD = 3  # debut in >= 3 national charts in the same week = coordinated global launch


def primary_artist(artists: str) -> str:
    if not isinstance(artists, str) or not artists.strip():
        return ""
    return artists.split("|")[0].split(",")[0].strip().lower()


def main():
    con = duckdb.connect()
    raw = con.sql(f"""
        SELECT key, title, artists, COUNT(*) AS n
        FROM read_parquet('{PARQUET_GLOB}')
        GROUP BY ALL
    """).df()
    # one representative title/artist per key (most frequent spelling)
    rep = raw.sort_values("n", ascending=False).drop_duplicates("key")
    ids = [normalize_song_identity(t, a) for t, a in zip(rep["title"], rep["artists"])]
    rep["song_id"] = [i[0] for i in ids]
    # keys with no usable title/artist keep their own identity rather than collapsing together
    empty = rep["title"].isna() | (rep["title"].fillna("").str.strip() == "")
    rep.loc[empty, "song_id"] = "k_" + rep.loc[empty, "key"]
    rep["primary_artist"] = rep["artists"].map(primary_artist)
    key_map = rep[["key", "song_id"]]
    key_map.to_parquet(INTERIM / "key_map.parquet", index=False)

    con.register("km", key_map)
    con.execute(f"""
        CREATE TABLE charts AS
        SELECT km.song_id, r.cc, CAST(r.week_end AS DATE) AS week_end,
               MIN(r.rank) AS rank, MAX(r.views) AS views
        FROM read_parquet('{PARQUET_GLOB}') r JOIN km USING (key)
        GROUP BY ALL
    """)
    con.execute(f"COPY charts TO '{(INTERIM / 'charts.parquet').as_posix()}' (FORMAT parquet)")

    # song table: representative metadata = the key with most rows inside the song
    meta = rep.sort_values("n", ascending=False).drop_duplicates("song_id")[
        ["song_id", "title", "artists", "primary_artist"]]
    con.register("meta", meta)
    songs = con.sql(f"""
        WITH nat AS (SELECT * FROM charts WHERE cc <> 'global'),
        first AS (SELECT song_id, MIN(week_end) AS debut_week FROM nat GROUP BY song_id),
        debut AS (
            SELECT n.song_id, f.debut_week, COUNT(DISTINCT n.cc) AS debut_markets,
                   LIST(DISTINCT n.cc ORDER BY n.cc) AS debut_ccs
            FROM nat n JOIN first f ON n.song_id = f.song_id AND n.week_end = f.debut_week
            GROUP BY ALL
        ),
        tot AS (SELECT song_id, COUNT(DISTINCT cc) AS total_markets FROM nat GROUP BY song_id)
        SELECT m.*, d.debut_week, d.debut_markets, d.debut_ccs, t.total_markets,
               d.debut_markets >= {SIMUL_THRESHOLD} AS is_simul_launch
        FROM debut d JOIN meta m USING (song_id) JOIN tot t USING (song_id)
    """).df()
    songs.to_parquet(INTERIM / "songs.parquet", index=False)

    merged = key_map.groupby("song_id").size()
    print(f"raw keys: {len(key_map):,} | canonical songs: {key_map.song_id.nunique():,} | "
          f"songs merged from >1 key: {(merged > 1).sum():,} ({int((merged[merged > 1] - 1).sum()):,} redundant keys)")
    print(f"songs with a national-chart debut: {len(songs):,}; simultaneous launches: {songs.is_simul_launch.sum():,} "
          f"({songs.is_simul_launch.mean():.2%})")
    print(con.sql("SELECT COUNT(*) rows, COUNT(DISTINCT cc) ccs, MIN(week_end), MAX(week_end) FROM charts").df())


if __name__ == "__main__":
    main()

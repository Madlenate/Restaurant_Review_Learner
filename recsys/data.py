"""
Stage 1 of the recommender pipeline: pull the modelling data out of the Yelp
DuckDB databases and write it to Parquet so every later step (NLP, CF, hybrid)
loads the same frozen snapshot instead of re-querying 7 GB of DuckDB.

What it writes (into ./extracts/)
---------------------------------
    rec_reviews.parquet    one row per review on an OPEN RESTAURANT
        review_id, user_id, business_id, stars, useful, funny, cool,
        date, text, n_tokens, split           (~3.8M rows, text included)

    rec_business.parquet   one row per open restaurant that has >=1 review
        business_id, name, city, state, latitude, longitude,
        business_avg_stars, business_review_count, price, categories,
        checkin_count, n_reviews_in_window

    rec_users.parquet      one row per user that has >=1 restaurant review
        user_id, user_name, user_avg_stars, user_review_count, fans,
        yelping_since, elite_years, n_rest_reviews, n_train_reviews

Train / test split
------------------
Purely temporal: every review strictly before CUTOFF is `train`, everything on
or after it is `test`.  A global date cut (not per-user leave-last-out) is the
honest setup here -- at prediction time you only ever know the past.

    CUTOFF = 2021-01-01   ->  ~89% train / ~11% test

Override with the env var  REC_SPLIT_CUTOFF=YYYY-MM-DD.

Usage
-----
    python -m recsys.data                 # build anything missing
    python -m recsys.data --force         # rebuild
    python -m recsys.data --peek          # just print stats, write nothing
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "yelp_dbs"
OUT_DIR = PROJECT_ROOT / "extracts"

CUTOFF = os.environ.get("REC_SPLIT_CUTOFF", "2021-01-01")

REVIEWS_PATH = OUT_DIR / "rec_reviews.parquet"
BUSINESS_PATH = OUT_DIR / "rec_business.parquet"
USERS_PATH = OUT_DIR / "rec_users.parquet"

# Restaurant filter reused everywhere.
REST = "b.categories ILIKE '%Restaurant%' AND b.is_open = 1"


def _connect() -> duckdb.DuckDBPyConnection:
    """In-memory connection with every Yelp db attached read-only."""
    con = duckdb.connect(":memory:")
    for name in ("business", "review", "users", "tip", "checkin"):
        p = (DB_DIR / f"{name}.duckdb").as_posix()
        con.execute(f"ATTACH '{p}' AS {name} (READ_ONLY)")
    con.execute(f"SET temp_directory='{(OUT_DIR / '.tmp').as_posix()}'")
    return con


def _build_reviews(con: duckdb.DuckDBPyConnection) -> None:
    print(f"  rec_reviews.parquet  (split cutoff {CUTOFF}) ...", flush=True)
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            SELECT
                r.review_id,
                r.user_id,
                r.business_id,
                r.stars,
                r.useful, r.funny, r.cool,
                r.date,
                r.text,
                len(string_split_regex(trim(r.text), '\\s+'))      AS n_tokens,
                CASE WHEN r.date < TIMESTAMP '{CUTOFF}'
                     THEN 'train' ELSE 'test' END                  AS split
            FROM review.review   AS r
            JOIN business.business AS b ON b.business_id = r.business_id
            WHERE {REST}
        ) TO '{REVIEWS_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION zstd)
        """
    )
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{REVIEWS_PATH.as_posix()}')"
    ).fetchone()[0]
    mb = REVIEWS_PATH.stat().st_size / 1024**2
    print(f"    {n:,} rows, {mb:,.0f} MB, {time.time() - t0:.0f}s", flush=True)


def _build_business(con: duckdb.DuckDBPyConnection) -> None:
    print("  rec_business.parquet ...", flush=True)
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            WITH chk AS (
                SELECT business_id,
                       len(string_split(date, ', ')) AS checkin_count
                FROM checkin.checkin
            ),
            rc AS (
                SELECT r.business_id, count(*) AS n_reviews_in_window
                FROM review.review AS r
                JOIN business.business AS b ON b.business_id = r.business_id
                WHERE {REST}
                GROUP BY 1
            )
            SELECT
                b.business_id,
                b.name,
                b.city,
                b.state,
                b.latitude,
                b.longitude,
                b.stars        AS business_avg_stars,
                b.review_count AS business_review_count,
                TRY_CAST(b.attributes->>'$.RestaurantsPriceRange2' AS INTEGER) AS price,
                b.categories,
                coalesce(chk.checkin_count, 0) AS checkin_count,
                rc.n_reviews_in_window
            FROM business.business AS b
            JOIN rc            ON rc.business_id  = b.business_id
            LEFT JOIN chk      ON chk.business_id = b.business_id
            WHERE {REST}
        ) TO '{BUSINESS_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION zstd)
        """
    )
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{BUSINESS_PATH.as_posix()}')"
    ).fetchone()[0]
    print(f"    {n:,} restaurants, {time.time() - t0:.0f}s", flush=True)


def _build_users(con: duckdb.DuckDBPyConnection) -> None:
    print("  rec_users.parquet ...", flush=True)
    t0 = time.time()
    con.execute(
        f"""
        COPY (
            WITH rr AS (
                SELECT
                    r.user_id,
                    count(*)                                        AS n_rest_reviews,
                    count(*) FILTER (WHERE r.date < TIMESTAMP '{CUTOFF}') AS n_train_reviews
                FROM review.review AS r
                JOIN business.business AS b ON b.business_id = r.business_id
                WHERE {REST}
                GROUP BY 1
            )
            SELECT
                u.user_id,
                u.name          AS user_name,
                u.average_stars AS user_avg_stars,
                u.review_count  AS user_review_count,
                u.fans,
                u.yelping_since,
                CASE WHEN u.elite IS NULL OR u.elite = '' THEN 0
                     ELSE len(string_split(u.elite, ',')) END       AS elite_years,
                rr.n_rest_reviews,
                rr.n_train_reviews
            FROM rr
            JOIN users.users AS u ON u.user_id = rr.user_id
        ) TO '{USERS_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION zstd)
        """
    )
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{USERS_PATH.as_posix()}')"
    ).fetchone()[0]
    print(f"    {n:,} users, {time.time() - t0:.0f}s", flush=True)


def _peek(con: duckdb.DuckDBPyConnection) -> None:
    pd.set_option("display.width", 200)
    print(f"\nsplit cutoff: {CUTOFF}\n")
    print(
        con.execute(
            f"""
            SELECT
                CASE WHEN r.date < TIMESTAMP '{CUTOFF}' THEN 'train' ELSE 'test' END AS split,
                count(*)                      AS reviews,
                count(DISTINCT r.user_id)     AS users,
                count(DISTINCT r.business_id) AS restaurants,
                round(avg(r.stars), 3)        AS avg_stars,
                min(r.date)::DATE             AS first,
                max(r.date)::DATE             AS last
            FROM review.review AS r
            JOIN business.business AS b ON b.business_id = r.business_id
            WHERE {REST}
            GROUP BY 1 ORDER BY 1
            """
        ).df()
    )
    print("\ncold users in test (no train review):")
    print(
        con.execute(
            f"""
            WITH t AS (
                SELECT DISTINCT r.user_id FROM review.review r
                JOIN business.business b ON b.business_id = r.business_id
                WHERE {REST} AND r.date >= TIMESTAMP '{CUTOFF}'
            ),
            tr AS (
                SELECT DISTINCT r.user_id FROM review.review r
                JOIN business.business b ON b.business_id = r.business_id
                WHERE {REST} AND r.date < TIMESTAMP '{CUTOFF}'
            )
            SELECT
                count(*)                                       AS test_users,
                count(*) FILTER (WHERE tr.user_id IS NULL)      AS cold_users
            FROM t LEFT JOIN tr USING (user_id)
            """
        ).df()
    )


# ---------------------------------------------------------------- public loader


def load(
    *,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reviews, business) DataFrames from the Parquet extracts.

    Build them first with `python -m recsys.data` if they are missing.
    """
    missing = [p.name for p in (REVIEWS_PATH, BUSINESS_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing extract(s): {missing} -- run `python -m recsys.data` first"
        )
    reviews = pd.read_parquet(REVIEWS_PATH, columns=columns)
    business = pd.read_parquet(BUSINESS_PATH)
    return reviews, business


def load_users() -> pd.DataFrame:
    if not USERS_PATH.exists():
        raise FileNotFoundError("run `python -m recsys.data` first")
    return pd.read_parquet(USERS_PATH)


# ------------------------------------------------------------------------ CLI


def main(argv: list[str]) -> int:
    force = "--force" in argv
    peek_only = "--peek" in argv

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / ".tmp").mkdir(exist_ok=True)

    con = _connect()
    try:
        if peek_only:
            _peek(con)
            return 0

        targets = [
            (REVIEWS_PATH, _build_reviews),
            (BUSINESS_PATH, _build_business),
            (USERS_PATH, _build_users),
        ]
        for path, fn in targets:
            if path.exists() and not force:
                print(f"  -- {path.name} exists (use --force to rebuild)")
                continue
            fn(con)

        print()
        _peek(con)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

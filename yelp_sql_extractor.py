"""
Query helper for the per-dataset Yelp DuckDB databases built by
`build_yelp_databases.py`.

DuckDB is an in-process ("embedded") engine, so there is no separate server to
start -- you just open the database file(s). Each Yelp dataset lives in its own
`.duckdb` file under ./yelp_dbs/. This module lets you either:

  * open one dataset:                 con = connect("review")
  * open all of them at once (ATTACH): con = connect_all()
    -> then every dataset is a schema:  review.review, business.business,
       checkin.checkin, tip.tip, users.users

Typical use
-----------
    from yelp_sql_extractor import connect_all, query, extract

    # cross-dataset join, straight to a DataFrame
    df = query('''
        SELECT b.name, b.city, count(*) AS n_reviews, avg(r.stars) AS avg_stars
        FROM review.review  r
        JOIN business.business b USING (business_id)
        WHERE b.city = 'New Orleans'
        GROUP BY 1, 2
        ORDER BY n_reviews DESC
        LIMIT 20
    ''')

    # pull a filtered slice out to Parquet for downstream work
    extract('''
        SELECT * FROM review.review
        WHERE date >= DATE '2021-01-01'
    ''', "extracts/reviews_2021.parquet")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / "yelp_dbs"

# dataset key -> (duckdb file, attach alias / schema name)
DATASETS: dict[str, tuple[str, str]] = {
    "business": ("business.duckdb", "business"),
    "checkin": ("checkin.duckdb", "checkin"),
    "review": ("review.duckdb", "review"),
    "tip": ("tip.duckdb", "tip"),
    "users": ("users.duckdb", "users"),
}


def db_path(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; valid: {list(DATASETS)}")
    p = DB_DIR / DATASETS[dataset][0]
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run `python build_yelp_databases.py {dataset}` first"
        )
    return p


def connect(dataset: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a single dataset's database."""
    return duckdb.connect(str(db_path(dataset)), read_only=read_only)


def connect_all(
    datasets: Iterable[str] | None = None,
    read_only: bool = True,
) -> duckdb.DuckDBPyConnection:
    """
    Open an in-memory connection with every dataset ATTACHed as its own schema.
    Query them as `<schema>.<table>`, e.g. `review.review`, `business.business`.
    """
    datasets = list(datasets) if datasets is not None else list(DATASETS)
    con = duckdb.connect(":memory:")
    ro = " (READ_ONLY)" if read_only else ""
    for ds in datasets:
        path = db_path(ds).as_posix()
        alias = DATASETS[ds][1]
        con.execute(f"ATTACH '{path}' AS {alias}{ro}")
    return con


def query(sql: str, *, datasets: Iterable[str] | None = None, params=None):
    """Run `sql` against all (or the given) datasets and return a DataFrame."""
    con = connect_all(datasets)
    try:
        rel = con.execute(sql, params) if params is not None else con.execute(sql)
        return rel.df()
    finally:
        con.close()


def extract(
    sql: str,
    out_path: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    params=None,
) -> Path:
    """
    Runs the whole extraction using sql and storing the path into the data 
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower().lstrip(".") or "parquet"

    fmt = {
        "parquet": "(FORMAT PARQUET)",
        "csv": "(FORMAT CSV, HEADER)",
        "json": "(FORMAT JSON)",
    }.get(ext)
    if fmt is None:
        raise ValueError(f"unsupported output extension: {out_path.suffix!r}")

    con = connect_all(datasets)
    try:
        if params is not None:
            con.execute(sql, params)
            con.execute(
                f"COPY (SELECT * FROM ({sql})) TO '{out_path.as_posix()}' {fmt}",
                params,
            )
        else:
            con.execute(f"COPY ({sql}) TO '{out_path.as_posix()}' {fmt}")
    finally:
        con.close()
    return out_path


if __name__ == "__main__":
    # this is used to make sure the tables were properly created and contain the data as needed
    for ds in DATASETS:
        try:
            c = connect(ds)
            tbl = DATASETS[ds][1]
            n = c.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
            cols = [r[0] for r in c.execute(f"DESCRIBE {tbl}").fetchall()]
            c.close()
            print(f"{ds:10s} {n:>12,} rows  |  {', '.join(cols)}")
        except (FileNotFoundError, duckdb.Error) as e:
            print(f"{ds:10s} (not available) {e}")

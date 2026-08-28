"""
Build one DuckDB database per Yelp Academic Dataset file.

The raw JSON is ~9 GB total (review.json alone is 5 GB), which is painful to load
into pandas. DuckDB reads the newline-delimited JSON in a streaming fashion and
stores it as a compressed columnar table on disk, so afterwards you can run SQL
against it and only pull the rows/columns you actually need.

Output (one file per dataset, in ./yelp_dbs/):
    business.duckdb   table: business
    checkin.duckdb    table: checkin
    review.duckdb     table: review
    tip.duckdb        table: tip
    users.duckdb      table: users

Usage:
    python build_yelp_databases.py                # build everything that's missing
    python build_yelp_databases.py review tip     # build only these
    python build_yelp_databases.py --force        # rebuild even if the .duckdb exists

Tuning (optional env vars):
    YELP_DB_MEMORY_LIMIT   default "6GB"
    YELP_DB_THREADS        default = all cores
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_ROOT / "Yelp_data_set" / "Yelp JSON" / "yelp_dataset"
OUTPUT_DIR = PROJECT_ROOT / "yelp_dbs"

TS_FORMAT = "%Y-%m-%d %H:%M:%S"  # Yelp timestamps: "2012-05-18 02:17:21" **EXAMPLE**


DATASETS: dict[str, dict] = {
    "business": {
        "db": "business.duckdb",
        "table": "business",
        "json": "yelp_academic_dataset_business.json",
        "columns": {
            "business_id": "VARCHAR",
            "name": "VARCHAR",
            "address": "VARCHAR",
            "city": "VARCHAR",
            "state": "VARCHAR",
            "postal_code": "VARCHAR",
            "latitude": "DOUBLE",
            "longitude": "DOUBLE",
            "stars": "DOUBLE",
            "review_count": "BIGINT",
            "is_open": "INTEGER",
            "attributes": "JSON",
            "categories": "VARCHAR",
            "hours": "JSON",
        },
    },
    "checkin": {
        "db": "checkin.duckdb",
        "table": "checkin",
        "json": "yelp_academic_dataset_checkin.json",
        # `date` is a single comma-separated string of timestamps -> keep as text
        "columns": {
            "business_id": "VARCHAR",
            "date": "VARCHAR",
        },
    },
    "review": {
        "db": "review.duckdb",
        "table": "review",
        "json": "yelp_academic_dataset_review.json",
        "columns": {
            "review_id": "VARCHAR",
            "user_id": "VARCHAR",
            "business_id": "VARCHAR",
            "stars": "DOUBLE",
            "useful": "BIGINT",
            "funny": "BIGINT",
            "cool": "BIGINT",
            "text": "VARCHAR",
            "date": "TIMESTAMP",
        },
    },
    "tip": {
        "db": "tip.duckdb",
        "table": "tip",
        "json": "yelp_academic_dataset_tip.json",
        "columns": {
            "user_id": "VARCHAR",
            "business_id": "VARCHAR",
            "text": "VARCHAR",
            "date": "TIMESTAMP",
            "compliment_count": "BIGINT",
        },
    },
    "users": {
        "db": "users.duckdb",
        "table": "users",  # "user" is a reserved-ish word in SQL, so "users"
        "json": "yelp_academic_dataset_user.json",
        "columns": {
            "user_id": "VARCHAR",
            "name": "VARCHAR",
            "review_count": "BIGINT",
            "yelping_since": "TIMESTAMP",
            "useful": "BIGINT",
            "funny": "BIGINT",
            "cool": "BIGINT",
            "elite": "VARCHAR",
            "friends": "VARCHAR",
            "fans": "BIGINT",
            "average_stars": "DOUBLE",
            "compliment_hot": "BIGINT",
            "compliment_more": "BIGINT",
            "compliment_profile": "BIGINT",
            "compliment_cute": "BIGINT",
            "compliment_list": "BIGINT",
            "compliment_note": "BIGINT",
            "compliment_plain": "BIGINT",
            "compliment_cool": "BIGINT",
            "compliment_funny": "BIGINT",
            "compliment_writer": "BIGINT",
            "compliment_photos": "BIGINT",
        },
    },
}


def _columns_struct(columns: dict[str, str]) -> str:
    inner = ", ".join(f"'{name}': '{typ}'" for name, typ in columns.items())
    return "{" + inner + "}"


def build_one(key: str, force: bool) -> None:
    spec = DATASETS[key]
    src = SOURCE_DIR / spec["json"]
    dst = OUTPUT_DIR / spec["db"]
    table = spec["table"]

    if not src.exists():
        print(f"  !! source missing, skipping: {src}")
        return

    if dst.exists() and not force:
        print(f"  -- {dst.name} already exists (use --force to rebuild)")
        return

    if dst.exists():
        dst.unlink()

    mem = os.environ.get("YELP_DB_MEMORY_LIMIT", "6GB")
    threads = os.environ.get("YELP_DB_THREADS")

    src_gb = src.stat().st_size / 1024**3
    print(f"  building {dst.name}  (source {src_gb:.1f} GB) ...", flush=True)
    t0 = time.time()

    con = duckdb.connect(str(dst))
    try:
        con.execute(f"SET memory_limit='{mem}'")
        con.execute("SET preserve_insertion_order=false")  # lower memory on big loads
        con.execute(f"SET temp_directory='{(OUTPUT_DIR / '.tmp').as_posix()}'")
        if threads:
            con.execute(f"SET threads={int(threads)}")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT *
            FROM read_json(
                '{src.as_posix()}',
                format = 'newline_delimited',
                columns = {_columns_struct(spec["columns"])},
                timestampformat = '{TS_FORMAT}'
            )
            """
        )
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        con.close()

    size_gb = dst.stat().st_size / 1024**3
    print(
        f"  done {dst.name}: {n:,} rows, {size_gb:.2f} GB on disk, "
        f"{time.time() - t0:,.0f}s",
        flush=True,
    )


def main(argv: list[str]) -> int:
    force = "--force" in argv
    wanted = [a for a in argv if not a.startswith("-")]
    keys = wanted or list(DATASETS)

    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {unknown}\nvalid: {list(DATASETS)}")
        return 2

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / ".tmp").mkdir(exist_ok=True)

    print(f"source:  {SOURCE_DIR}")
    print(f"output:  {OUTPUT_DIR}")
    print(f"datasets: {keys}\n")

    for key in keys:
        build_one(key, force)

    print("\nall requested databases built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

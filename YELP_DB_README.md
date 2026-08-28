# Yelp dataset → DuckDB

The raw Yelp Academic Dataset (`Yelp_data_set/Yelp JSON/yelp_dataset/*.json`) is
~9 GB of newline-delimited JSON — too big to comfortably load with pandas. This
setup loads each file into its own on-disk **DuckDB** database so you can query it
with SQL and only pull the rows/columns you actually need.

> DuckDB is an *embedded* engine (like SQLite). There is no server process to
> start or manage — you just open the `.duckdb` file(s).

## Layout

```
build_yelp_databases.py   # JSON  ->  yelp_dbs/*.duckdb   (run once)
yelp_sql_extractor.py      # query helpers (connect / connect_all / query / extract)
yelp_dbs/
    business.duckdb   table: business    150,346 rows
    checkin.duckdb    table: checkin     131,930 rows
    review.duckdb     table: review    ~6,990,280 rows
    tip.duckdb        table: tip         908,915 rows
    users.duckdb      table: users    ~1,987,000 rows
```

Nested objects `business.attributes` and `business.hours` are stored as `JSON`
columns — query them with DuckDB's JSON operators, e.g.
`attributes->>'$.RestaurantsPriceRange2'`.

## Build

```bash
python build_yelp_databases.py                 # build any missing databases
python build_yelp_databases.py review users    # build specific ones
python build_yelp_databases.py --force         # rebuild everything
```

Env knobs: `YELP_DB_MEMORY_LIMIT` (default `6GB`), `YELP_DB_THREADS`.

## Query

```python
from yelp_sql_extractor import connect, connect_all, query, extract

# a) one dataset
con = connect("review")
con.sql("SELECT stars, count(*) FROM review GROUP BY 1 ORDER BY 1").show()

# b) all datasets at once — each is a schema: business.business, review.review, ...
df = query('''
    SELECT b.city, count(*) n_reviews, avg(r.stars) avg_stars
    FROM review.review r
    JOIN business.business b USING (business_id)
    GROUP BY 1 ORDER BY n_reviews DESC LIMIT 10
''')

# c) extract a filtered slice to Parquet/CSV for downstream work
extract('''
    SELECT * FROM review.review WHERE date >= DATE '2021-01-01'
''', "extracts/reviews_2021.parquet")
```

### Same thing from the DuckDB CLI / any other tool

```sql
ATTACH 'yelp_dbs/review.duckdb'     AS review     (READ_ONLY);
ATTACH 'yelp_dbs/business.duckdb'   AS business   (READ_ONLY);
SELECT * FROM review.review LIMIT 5;
```

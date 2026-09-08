"""
Precompute the small, fast-loading artifacts the Streamlit app needs, so the app
never has to touch the 900 MB rec_reviews.parquet at request time.

Inputs
------
    models/als.npz              (from the notebook ALS cell)
    models/seen.parquet         (from the notebook: tr[["user_id","business_id"]])
    extracts/rec_reviews.parquet
    extracts/rec_business.parquet
    extracts/rec_users.parquet

Outputs (models/app/)
---------------------
    history.parquet     one row per (warm user, restaurant) review:
                        user_id, business_id, stars, date, useful, split
    snippets.parquet    one representative review per restaurant:
                        business_id, snippet, stars, useful
    user_home.parquet   user_id, home_city, home_state, n_home
    users.parquet       warm-user profile slice from rec_users

Run:  python -m scripts.build_app_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXTRACTS = ROOT / "extracts"
MODELS = ROOT / "models"
OUT = MODELS / "app"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    als = np.load(MODELS / "als.npz", allow_pickle=True)
    warm_users = set(als["user_ids"].tolist())
    catalog = set(als["item_ids"].tolist())
    print(f"warm users {len(warm_users):,}   catalog {len(catalog):,}")

    biz = pd.read_parquet(EXTRACTS / "rec_business.parquet")

    # ---- history: every review by a warm user on a catalog restaurant ----
    rv = pd.read_parquet(
        EXTRACTS / "rec_reviews.parquet",
        columns=["user_id", "business_id", "stars", "date", "useful", "split"],
    )
    hist = rv[rv.user_id.isin(warm_users) & rv.business_id.isin(catalog)].copy()
    hist.to_parquet(OUT / "history.parquet", index=False)
    print(f"history.parquet     {len(hist):,} rows")

    # ---- user_home: modal city/state of the restaurants each user reviewed ----
    hb = hist.merge(biz[["business_id", "city", "state"]], on="business_id", how="left")
    home = (
        hb.groupby(["user_id", "city", "state"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False)
        .drop_duplicates("user_id")
        .rename(columns={"city": "home_city", "state": "home_state", "n": "n_home"})
    )
    home.to_parquet(OUT / "user_home.parquet", index=False)
    print(f"user_home.parquet   {len(home):,} rows")

    # ---- snippets: one useful, positive review per catalog restaurant ----
    txt = pd.read_parquet(
        EXTRACTS / "rec_reviews.parquet",
        columns=["business_id", "stars", "useful", "text", "split", "n_tokens"],
    )
    cand = txt[
        txt.business_id.isin(catalog)
        & (txt.split == "train")
        & (txt.stars >= 4)
        & (txt.n_tokens.between(20, 120))
    ].copy()
    cand = cand.sort_values("useful", ascending=False).drop_duplicates("business_id")
    cand["snippet"] = cand["text"].str.slice(0, 400).str.replace(r"\s+", " ", regex=True)
    cand[["business_id", "snippet", "stars", "useful"]].to_parquet(
        OUT / "snippets.parquet", index=False
    )
    print(f"snippets.parquet    {len(cand):,} rows "
          f"({len(cand) / len(catalog):.0%} of catalog)")

    # ---- users: profile slice for the warm users ----
    users = pd.read_parquet(EXTRACTS / "rec_users.parquet")
    users = users[users.user_id.isin(warm_users)]
    users.to_parquet(OUT / "users.parquet", index=False)
    print(f"users.parquet       {len(users):,} rows")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

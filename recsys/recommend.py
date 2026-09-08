"""
Recommender core for the Streamlit demo.

The trained model is just the two ALS factor matrices; a recommendation is a
dot product, a mask for already-visited restaurants, an optional city filter,
and a top-N sort. No `implicit` dependency at serve time -- numpy only.

    from recsys.recommend import Recommender
    rec = Recommender.load()
    uid = rec.random_user()
    rec.history(uid)               # what this user has reviewed
    rec.recommend(uid, n=5)        # top-5 restaurants, same city, unseen
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
APP = MODELS / "app"


@dataclass
class Recommender:
    U: np.ndarray                 # (n_users, f) user factors
    V: np.ndarray                 # (n_items, f) item factors
    user_ids: np.ndarray          # row index -> user_id
    item_ids: np.ndarray          # row index -> business_id
    business: pd.DataFrame         # catalog metadata, indexed by business_id
    history_df: pd.DataFrame       # all warm-user reviews
    user_home: pd.DataFrame        # user_id -> home_city / home_state
    users: pd.DataFrame            # warm-user profiles, indexed by user_id
    snippets: pd.DataFrame         # business_id -> representative review

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls) -> "Recommender":
        als = np.load(MODELS / "als.npz", allow_pickle=True)
        biz = (
            pd.read_parquet("extracts/rec_business.parquet")
            if (ROOT / "extracts/rec_business.parquet").exists()
            else pd.read_parquet(ROOT / "extracts/rec_business.parquet")
        ).set_index("business_id")
        users = pd.read_parquet(APP / "users.parquet").set_index("user_id")
        return cls(
            U=als["user_factors"],
            V=als["item_factors"],
            user_ids=als["user_ids"],
            item_ids=als["item_ids"],
            business=biz,
            history_df=pd.read_parquet(APP / "history.parquet"),
            user_home=pd.read_parquet(APP / "user_home.parquet").set_index("user_id"),
            users=users,
            snippets=pd.read_parquet(APP / "snippets.parquet").set_index("business_id"),
        )

    # ------------------------------------------------------------- id lookups

    @cached_property
    def _uidx(self) -> dict[str, int]:
        return {u: k for k, u in enumerate(self.user_ids)}

    @cached_property
    def _iidx(self) -> dict[str, int]:
        return {b: k for k, b in enumerate(self.item_ids)}

    @cached_property
    def _seen(self) -> dict[str, set[str]]:
        return self.history_df.groupby("user_id").business_id.agg(set).to_dict()

    # ---------------------------------------------------------------- queries

    def random_user(self, rng: np.random.Generator | None = None,
                    min_reviews: int = 5) -> str:
        rng = rng or np.random.default_rng()
        counts = self.history_df.groupby("user_id").size()
        pool = counts[counts >= min_reviews].index.to_numpy()
        return str(rng.choice(pool))

    def profile(self, user_id: str) -> dict:
        u = self.users.loc[user_id] if user_id in self.users.index else None
        home = (
            self.user_home.loc[user_id]
            if user_id in self.user_home.index
            else None
        )
        h = self.history_df[self.history_df.user_id == user_id]
        return {
            "user_id": user_id,
            "name": None if u is None else u.get("user_name"),
            "n_reviews": int(len(h)),
            "avg_stars": float(h.stars.mean()) if len(h) else None,
            "home_city": None if home is None else home["home_city"],
            "home_state": None if home is None else home["home_state"],
            "yelping_since": None if u is None else u.get("yelping_since"),
            "elite_years": None if u is None else int(u.get("elite_years", 0) or 0),
        }

    def history(self, user_id: str, n: int | None = None) -> pd.DataFrame:
        h = self.history_df[self.history_df.user_id == user_id].copy()
        h = h.join(self.business[["name", "city", "categories", "price"]],
                   on="business_id")
        h = h.sort_values(["stars", "date"], ascending=[False, False])
        return h.head(n) if n else h

    def recommend(
        self,
        user_id: str,
        n: int = 5,
        same_city: bool = True,
        min_business_reviews: int = 20,
    ) -> pd.DataFrame:
        if user_id not in self._uidx:
            raise KeyError(f"{user_id!r} is not a warm user in the ALS model")

        scores = self.V @ self.U[self._uidx[user_id]]          # (n_items,)
        order = np.argsort(-scores)

        seen = self._seen.get(user_id, set())
        home_city = None
        if same_city and user_id in self.user_home.index:
            home_city = self.user_home.loc[user_id, "home_city"]

        out = []
        for k in order:
            bid = self.item_ids[k]
            if bid in seen or bid not in self.business.index:
                continue
            row = self.business.loc[bid]
            if home_city is not None and row["city"] != home_city:
                continue
            if row["business_review_count"] < min_business_reviews:
                continue
            out.append(
                {
                    "business_id": bid,
                    "name": row["name"],
                    "city": row["city"],
                    "state": row["state"],
                    "price": row["price"],
                    "avg_stars": row["business_avg_stars"],
                    "review_count": int(row["business_review_count"]),
                    "categories": row["categories"],
                    "match_score": float(scores[k]),
                    "snippet": (
                        self.snippets.loc[bid, "snippet"]
                        if bid in self.snippets.index
                        else None
                    ),
                }
            )
            if len(out) >= n:
                break
        return pd.DataFrame(out)

    # generic tags that carry no signal about taste
    _CAT_STOP = {"Restaurants", "Food", "Nightlife", "Bars", "Event Planning & Services"}

    def why(self, user_id: str, business_id: str) -> str:
        """One-line rationale from category overlap with the user's favourites."""
        h = self.history(user_id)
        liked = h[h.stars >= 4]
        user_cats: dict[str, int] = {}
        for c in liked["categories"].dropna():
            for t in [x.strip() for x in c.split(",")]:
                if t and t not in self._CAT_STOP:
                    user_cats[t] = user_cats.get(t, 0) + 1
        rec_cats = {
            x.strip()
            for x in (self.business.loc[business_id, "categories"] or "").split(",")
        } - self._CAT_STOP
        shared = sorted(rec_cats & user_cats.keys(), key=lambda t: -user_cats[t])[:3]
        if shared:
            return "matches your visits to " + ", ".join(shared)
        return "similar to restaurants you rated highly"

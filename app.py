"""
Restaurant recommender demo.

    streamlit run app.py

Draws a random Yelp reviewer who has a review history, shows what they've
reviewed, and recommends restaurants in their city using the ALS
collaborative-filtering model trained in Data_Analysis.ipynb.

Prereqs (build once):
    - run the ALS + save cells in the notebook   -> models/als.npz, models/seen.parquet
    - python -m scripts.build_app_data            -> models/app/*.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from recsys.recommend import Recommender

st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="wide")


@st.cache_resource
def get_recommender() -> Recommender:
    return Recommender.load()


rec = get_recommender()

# --------------------------------------------------------------------- sidebar

st.sidebar.header("Controls")
same_city = st.sidebar.toggle("Restrict to the user's city", value=True)
n_recs = st.sidebar.slider("Number of recommendations", 3, 10, 5)
min_rc = st.sidebar.slider("Min. reviews a restaurant must have", 0, 100, 20, step=5)

if st.sidebar.button("🎲  Draw a random reviewer", use_container_width=True):
    st.session_state["uid"] = rec.random_user()

st.sidebar.caption(
    "Model: implicit ALS on 1.7M train ratings, 133k users, 29k restaurants. "
    "Only users with a review history are eligible (no cold start here)."
)

# --------------------------------------------------------------------- header

st.title("🍽️ Restaurant Recommender")

if "uid" not in st.session_state:
    st.session_state["uid"] = rec.random_user()

uid = st.session_state["uid"]
p = rec.profile(uid)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviewer", p["name"] or uid[:8])
c2.metric("Reviews", p["n_reviews"])
c3.metric("Avg rating given", f"{p['avg_stars']:.2f}★" if p["avg_stars"] else "—")
c4.metric("Home city", p["home_city"] or "—")

if p["avg_stars"] and p["avg_stars"] < 3.4:
    st.warning(
        "This is a **harsh reviewer** (avg < 3.4★). The model predicts these "
        "users least well — see the limitations section in the notebook."
    )

st.caption(f"user_id `{uid}`  ·  elite years: {p['elite_years']}  ·  on Yelp since {p['yelping_since']}")

# --------------------------------------------------------------------- body

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("What they've reviewed")
    h = rec.history(uid)
    show = h[["name", "city", "stars", "date", "categories"]].rename(
        columns={"name": "restaurant", "stars": "★"}
    )
    st.dataframe(show, hide_index=True, use_container_width=True, height=430)

    counts = (
        h.assign(cat=h["categories"].str.split(", "))
        .explode("cat")
        .cat.value_counts()
        .head(8)
    )
    st.caption("Most-reviewed categories: " + ", ".join(counts.index))

with right:
    st.subheader(f"Top {n_recs} recommendations")
    recs = rec.recommend(uid, n=n_recs, same_city=same_city, min_business_reviews=min_rc)

    if recs.empty:
        st.info(
            "No candidates left after filtering. Try turning off the city "
            "restriction or lowering the min-reviews threshold."
        )
    else:
        for _, r in recs.iterrows():
            price = "$" * int(r["price"]) if pd.notna(r["price"]) else ""
            with st.container(border=True):
                st.markdown(f"### {r['name']}  \n{r['city']}, {r['state']}  ·  {price}")
                m1, m2 = st.columns(2)
                m1.metric("Yelp avg", f"{r['avg_stars']:.1f}★", f"{r['review_count']} reviews",
                          delta_color="off")
                m2.metric("Match score", f"{r['match_score']:.2f}")
                st.caption(r["categories"])
                st.write(f"**Why:** {rec.why(uid, r['business_id'])}")
                if r["snippet"]:
                    st.markdown(f"> {r['snippet']}…")

st.divider()
with st.expander("How this works / caveats"):
    st.markdown(
        """
- **Recommendations** come from an implicit-feedback ALS model: each user and
  restaurant is a 64-dim vector, the score is their dot product. Restaurants the
  user already reviewed are removed; by default results are limited to the user's
  home city.
- **"Why"** is a simple category overlap with the user's highly-rated visits.
  The review snippet is the most up-voted 4–5★ review of that restaurant.
- **Not shown:** exact star-rating prediction (RMSE ≈ 1.16, near this dataset's
  floor) and cold-start users (61% of the test set) — the notebook covers why
  both are out of scope.
"""
    )

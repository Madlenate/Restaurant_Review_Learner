# Restaurant Recommender — Streamlit demo

Draws a random Yelp reviewer with a review history and recommends restaurants in
their city, using the ALS collaborative-filtering model from `Data_Analysis.ipynb`.

## Files

| path | what |
|---|---|
| `app.py` | the Streamlit UI |
| `recsys/recommend.py` | `Recommender` class — load model, `random_user`, `profile`, `history`, `recommend`, `why` |
| `scripts/build_app_data.py` | precomputes the small artifacts the app loads (`models/app/*.parquet`) |
| `recsys/data.py` | the frozen-extract loader (already built) |

## Build the artifacts (once)

1. In `Data_Analysis.ipynb`, run the ALS cell and the save cell:
   ```python
   np.savez_compressed("models/als.npz", user_factors=U, item_factors=V,
                       user_ids=users_ids, item_ids=item_ids)
   tr[["user_id", "business_id"]].to_parquet("models/seen.parquet", index=False)
   ```
2. Build the app data:
   ```
   python -m scripts.build_app_data
   ```
   → writes `models/app/{history,snippets,user_home,users}.parquet`

## Run

```
streamlit run app.py
```

## What it does / doesn't

- **Does:** rank restaurants for users with a review history (implicit ALS, 64-dim
  factors, dot-product score), filter to the user's home city, drop already-visited
  places, explain each pick via category overlap + a representative review.
- **Doesn't:** predict exact star ratings (RMSE ≈ 1.16, near this dataset's floor),
  or serve cold-start users (61% of the test set) — see the limitations section in
  the notebook.

## Where to take it next

- Wire real aspect sentiment into `Recommender.why()` (needs a full aspect pass over
  the catalog's reviews — currently `why()` uses category overlap only).
- Show the bias-model predicted rating next to each rec (save `mu, b_u, b_i`).
- Add a map (`st.map` on `latitude` / `longitude` from `rec_business.parquet`).
- Let the user pick a city / cuisine instead of only a random draw.

# Yelp Restaurant Recommender

A restaurant recommendation system built on the Yelp Open Dataset (6M+ reviews),
combining **NLP sentiment/aspect analysis** with **collaborative filtering** to
predict ratings and rank restaurant recommendations for users.

---

## 🔍 TL;DR

- Median user has written only **~3 restaurant reviews** → the user×restaurant
  matrix is **~0.045% dense**.
- Because of that sparsity, rating-prediction models (bias model, matrix
  factorization, SVD++) all plateau around **RMSE ≈ 1.16** (about one star off) —
  the fancier models barely beat the simplest one.
- **61% of test users are cold-start** (no prior review history), so
  collaborative filtering can't score them at all.
- What *does* work well: a **sentiment model** (TF-IDF + Logistic Regression)
  that separates positive from negative reviews with **ROC-AUC ≈ 0.99**, and an
  **aspect model** that explains *why* a restaurant is liked or disliked
  (food, service, price, ambiance, wait).
- **Conclusion:** the bottleneck is data density, not modeling technique. The
  most promising next step is using the NLP layer to generate content-based
  explanations and scorecards, rather than squeezing more out of collaborative
  filtering alone.

---

## 📊 App / Project Description *(reusable blurb)*

> Use this section as-is for a Streamlit `st.markdown()` about box, sidebar
> description, or app header.

**Yelp Restaurant Recommender** is a data science project that analyzes
millions of Yelp restaurant reviews to understand what makes people rate a
restaurant highly — and to recommend restaurants a user is likely to enjoy.
It combines two techniques: a **sentiment/aspect NLP model** that reads review
text to identify what people liked or disliked (food, service, price,
ambiance, wait time), and a **collaborative filtering system** (bias models,
matrix factorization, and implicit ALS) that learns from patterns in who
reviewed what. The app surfaces restaurant recommendations along with
plain-language explanations of *why* they were recommended, generated
directly from the aspect analysis.

---

## 🗂 Data

Sourced from the **Yelp Open Dataset**, queried with **DuckDB** (chosen for its
ability to handle the dataset's size without loading everything into memory).

| Table | Contents |
|---|---|
| `business` | Restaurant metadata, categories, star rating, open/closed status |
| `review` | ~6M reviews, star rating, review text, useful/funny/cool votes |
| `checkin` | Foot-traffic / check-in timestamps per business |
| `tip` | Short-form reviews / shout-outs |
| `users` | User profile stats: review count, fans, elite status, compliments |

Analysis is scoped to **open restaurants only** (`categories ILIKE '%Restaurant%' AND is_open = 1`).

---

## 🧪 Methodology

### 1. Exploratory Analysis
- Built a joined "review cube" across business, review, user, tip, and
  check-in tables.
- Found the median user review count (~3) — the key constraint on everything
  downstream.
- Deep-dived a real case (a consistently harsh reviewer, "George") to motivate
  aspect-level analysis: a low review on an otherwise well-loved restaurant
  isn't noise, it's a specific complaint about specific aspects.

**Review count distribution (log scale):**

<img width="590" height="490" alt="review_count_distribution" src="https://github.com/user-attachments/assets/3ae184e1-62fc-4a0b-af5e-a5fe74c8f5c4" />

*Most users have written 1–3 reviews; a long tail of "super-reviewers" pulls
the mean up but the matrix stays extremely sparse.*

### 2. Sentiment Model (NLP)
- Labeled reviews positive (4–5★) / negative (1–2★), dropped ambiguous 3★.
- Pipeline: `TfidfVectorizer` (unigrams) → `LogisticRegression`, trained on a
  strict time-based train/test split (train = pre-2021).
- Result: **ROC-AUC ≈ 0.99**, ~97–98% accuracy on each class.

**Held-out confusion matrix:**

<img width="446" height="390" alt="sentiment_confusion_matrix" src="https://github.com/user-attachments/assets/d4fea98f-ed32-444c-881b-dfe286e4afbd" />

### 3. Aspect Analysis
- Split reviews into sentences, scored each with the sentiment model, and
  tagged sentences by aspect keyword (food / service / price / ambiance / wait).
- A ridge regression on restaurant-level aspect profiles showed **food,
  ambiance, and short wait times** correlate with better ratings, while
  **price and service complaints** correlate with worse ones.
- Produces both a per-restaurant "aspect profile" and a per-user "aspect
  deviation" (which topics a given user is unusually harsh or generous about).

### 4. Collaborative Filtering
Three explicit-rating models trained on a strict train/test split, plus one
implicit ranking model:

| Model | Formula | Purpose |
|---|---|---|
| Damped bias model | `μ + b_u + b_i` | Baseline: global mean, item bias, user bias |
| SGD matrix factorization | `μ + b_u + b_i + p_u·q_i` | Learns latent taste vectors on top of the bias residual |
| SVD++ | `μ + b_u + b_i + q_i·(p_u + implicit term)` | Adds "which restaurants a user reviewed at all" as signal |
| Implicit **ALS** | user×item interaction matrix factorization | Top-N ranking / recommendation serving |

- Bias, MF, and SVD++ were evaluated on **RMSE / MAE** for rating prediction.
- **ALS** was evaluated on **Recall@10 / NDCG@10** for top-N ranking — this is
  the model actually used to serve recommendations, since ranking (not exact
  star prediction) is what a recommender app needs.

### 5. Sanity Checks
- Correlated aspect-based rating estimates, aspect-match scores, and LSA text
  similarity against true star ratings to confirm the content-based features
  carry real signal, independent of the CF models.

---

## ✅ Results Summary

| Metric | Result |
|---|---|
| Sentiment classification | ROC-AUC ≈ 0.99 |
| Bias / MF / SVD++ rating RMSE | ≈ 1.16 (all three converge to roughly the same floor) |
| Cold-start test users | 61% (no CF coverage) |
| User×restaurant matrix density | ~0.045% |
| ALS ranking | Recall@10 / NDCG@10 reported on warm users only |

---

## 🚀 Next Steps

1. Use the aspect model to generate **plain-language recommendation
   explanations** (e.g. *"Reviewers consistently praise the food and service
   here; a few flag slow waits."*).
2. Build a **scorecard UI** per restaurant showing food / service / price /
   ambiance / wait, driven by the aspect model.
3. Grow the review-per-user sample (more data, longer history) to reduce
   cold-start rate and give collaborative filtering more signal per user.

---

## 🛠 Tech Stack

- **DuckDB** — querying the large Yelp dataset
- **pandas / NumPy** — data wrangling
- **scikit-learn** — TF-IDF, Logistic Regression, Ridge, TruncatedSVD (LSA)
- **implicit** — Alternating Least Squares (ALS) for top-N ranking
- **matplotlib** — visualizations
- **Streamlit** *(planned/serving layer)* — interactive app for browsing
  recommendations and aspect scorecards

---

## 📁 Project Structure

```
├── Data_Analysis.ipynb        # Full analysis notebook (this project)
├── models/
│   ├── als.npz                # Saved ALS user/item factors
│   └── seen.parquet           # Train interactions (for filtering seen items)
├── review_count_distribution.png
├── sentiment_confusion_matrix.png
└── README.md
```

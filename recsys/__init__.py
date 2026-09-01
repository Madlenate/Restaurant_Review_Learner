"""Restaurant recommender: NLP + collaborative-filtering pipeline.

Modules
-------
    data   -- pull the modelling extracts out of the Yelp DuckDB databases
              and assign a temporal train/test split.

Typical use
-----------
    python -m recsys.data            # build extracts/  (run once)

    from recsys.data import load
    reviews, business = load()       # ready-to-model DataFrames
    train = reviews[reviews.split == "train"]
    test  = reviews[reviews.split == "test"]
"""

"""
Review sentiment classifier on PyroChain text features.

Shows:
- Batch feature extraction: ``PyroChain.extract_features`` on a list of records.
- Turning the returned tensors into a NumPy matrix for scikit-learn.
- A 5-fold cross-validated logistic-regression accuracy, compared with a majority-class baseline.

Data: ``examples/assets/reviews.csv``, 40 short product reviews written for this example
(20 positive, 20 negative). Needs: ``pip install pyrochain``. No API key. The first run
downloads ``microsoft/DialoGPT-small`` (about 350 MB) from the Hugging Face Hub; after that it
runs offline.

Run: python examples/review_sentiment_classifier.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pyrochain import PyroChain, PyroChainConfig

REVIEWS = Path(__file__).parent / "assets" / "reviews.csv"


def main() -> None:
    with REVIEWS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["label"]) for r in rows])

    chain = PyroChain(PyroChainConfig(device="cpu"))
    result = chain.extract_features(
        [{"text": r["text"]} for r in rows], "Extract features for review sentiment", validate=False
    )
    X = np.vstack([f["text_features"].cpu().numpy() for f in result["features"]])
    print(f"Feature matrix: {X.shape[0]} reviews x {X.shape[1]} dims ({chain.config.model_name})")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, labels, cv=folds)
    baseline = max(labels.mean(), 1 - labels.mean())
    print(f"5-fold accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")
    print(f"Majority-class baseline: {baseline:.2f}")

    model.fit(X, labels)
    for text in ("Arrived broken and support ignored me.", "Fantastic quality, works perfectly."):
        x = chain.extract_features({"text": text}, "sentiment", validate=False)
        prob = model.predict_proba(x["features"][0]["text_features"].cpu().numpy())[0, 1]
        print(f"P(positive) = {prob:.2f}  {text}")


if __name__ == "__main__":
    main()

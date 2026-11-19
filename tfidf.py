"""TF-IDF + logistic regression: B1 majority, B2 ERM, and the same mitigations as the transformer.

Variants: erm | erm80k (same 80K subsample as bert) | reweight | cda | soft | combo (= cda + reweight + soft).
Soft labels: each comment appears twice, labelled 1 with weight p and 0 with weight 1-p (p = annotator fraction).
"""
import sys, time, joblib
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import prepare, cda_copy, group_weights, TERM_RE, MODELS, PREDS


def train(variant, d, vec):
    rng = np.random.default_rng(0)
    tr = d[d.split == "train"]
    if variant == "erm80k":
        tr = tr.sample(80_000, random_state=0)
    tr = tr.reset_index(drop=True)
    if variant in ("cda", "combo"):
        has = tr[tr.text.map(lambda t: bool(TERM_RE.search(t)))]
        tr = pd.concat([tr, has.assign(text=[cda_copy(t, rng) for t in has.text])], ignore_index=True)
    w = group_weights(tr.group) if variant in ("reweight", "combo") else np.ones(len(tr))
    X = vec.transform(tr.text)
    if variant in ("soft", "combo"):
        p = tr.toxicity_soft.values
        X, y, w = X[np.r_[np.arange(len(tr)), np.arange(len(tr))]], np.r_[np.ones(len(tr)), np.zeros(len(tr))], np.r_[w * p, w * (1 - p)]
        keep = w > 0
        X, y, w = X[keep], y[keep], w[keep]
    else:
        y = tr.y.values
    t0 = time.time()
    m = LogisticRegression(C=4, max_iter=300, solver="liblinear").fit(X, y, sample_weight=w)
    joblib.dump((vec, m), MODELS / f"tfidf_{variant}.joblib")
    for split in ("val", "test"):
        s = d[d.split == split]
        pd.DataFrame({"toxicity": m.decision_function(vec.transform(s.text)), "id": s.id.values}).to_parquet(PREDS / f"tfidf_{variant}_{split}.parquet")
    print(f"tfidf_{variant}: {len(y)} rows, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True); PREDS.mkdir(exist_ok=True)
    d = prepare()
    for split in ("val", "test"):  # B1: majority class -> constant score (AUC 0.5, predicts all non-toxic)
        s = d[d.split == split]
        pd.DataFrame({"toxicity": np.full(len(s), -10.0), "id": s.id.values}).to_parquet(PREDS / f"majority_{split}.parquet")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True).fit(d.text[d.split == "train"])
    for v in sys.argv[1:] or ["erm", "erm80k", "reweight", "cda", "soft", "combo"]:
        train(v, d, vec)

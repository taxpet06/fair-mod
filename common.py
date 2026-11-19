"""Shared pieces: data prep, identity lexicon + swaps, groups, metrics, integrated gradients."""
import re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
DATA, PREDS, MODELS, RESULTS = (ROOT / d for d in ("data", "preds", "models", "results"))
IDS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]
LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]

# Identity terms swapped within a category. Used for training-time augmentation (CDA) and the live swap table.
SWAP = {
    "religion": ["muslim", "christian", "jewish", "hindu", "buddhist", "atheist"],
    "religion_pl": ["muslims", "christians", "jews", "hindus", "buddhists", "atheists"],
    "orientation": ["gay", "straight", "lesbian", "bisexual", "transgender"],
    "race": ["black", "white", "asian", "latino"],
    "gender": ["woman", "man"], "gender_pl": ["women", "men"],
}
TERM2CAT = {t: c for c, ts in SWAP.items() for t in ts}
TERM_RE = re.compile(r"\b(" + "|".join(sorted(TERM2CAT, key=len, reverse=True)) + r")\b", re.I)


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def _case_like(src, word):
    return word.capitalize() if src[:1].isupper() else word


def swap_terms(text, target_of):
    """Replace every lexicon term with target_of(term_lower) (or keep it if that returns None), preserving capitalisation."""
    return TERM_RE.sub(lambda m: _case_like(m.group(0), target_of(m.group(0).lower()) or m.group(0)), text)


def identity_variants(text):
    """For the first lexicon term in text, return {replacement: swapped_text} over its category (incl. original)."""
    m = TERM_RE.search(text)
    if not m:
        return {}
    src = m.group(0).lower()
    return {t: swap_terms(text, lambda w: t if w == src else None) for t in SWAP[TERM2CAT[src]]}


def cda_copy(text, rng):
    """One counterfactual copy: every lexicon term mapped to a random other member of its category."""
    pick = {}
    def tgt(w):
        if w not in pick:
            pick[w] = rng.choice([t for t in SWAP[TERM2CAT[w]] if t != w])
        return pick[w]
    return swap_terms(text, tgt)


def prepare():
    """Normalise, dedup, remove cross-split leakage, join soft labels, assign groups. Cached to data/prepared.parquet."""
    out = DATA / "prepared.parquet"
    if out.exists():
        return pd.read_parquet(out)
    w = pd.concat([pd.read_parquet(DATA / f"wilds/data/{s}-00000-of-00001.parquet").assign(split=s[:5].replace("valid", "val"))
                   for s in ("train", "validation", "test")], ignore_index=True)
    raw = pd.read_parquet(DATA / "wilds/raw/train-00000-of-00001.parquet", columns=["id", "toxicity_annotator_count"] + LABELS)
    w = w.drop(columns=LABELS).merge(raw.rename(columns={l: l + "_soft" for l in LABELS}), on="id", how="left")
    assert w.toxicity_soft.notna().all()
    w["text"] = w.comment_text.map(norm)
    w["y"] = (w.toxicity_soft >= 0.5).astype(int)
    n0 = w.split.value_counts()
    seen_test, seen_val = set(w.text[w.split == "test"]), set(w.text[w.split == "val"])
    leak = ((w.split != "test") & w.text.isin(seen_test)) | ((w.split == "train") & w.text.isin(seen_val))
    w = w[~leak]
    w = w[~w.duplicated(["split", "text"])]
    print("rows removed by leakage filter + dedup:", (n0 - w.split.value_counts()).to_dict())
    # group = rarest identity mentioned (or none) x label -> 18 groups, used by reweighting and group DRO
    freq = w.loc[w.split == "train", IDS].mean()
    order = list(freq.sort_values().index)
    w["idgrp"] = w[order].values.argmax(1)
    w.loc[w[IDS].sum(axis=1) == 0, "idgrp"] = len(IDS)
    w["idgrp"] = np.array(order + ["none"])[w.idgrp]
    w["group"] = w.idgrp + "_" + w.y.astype(str)
    w = w[["id", "split", "text", "y", "toxicity_annotator_count", "idgrp", "group"] + [l + "_soft" for l in LABELS] + IDS].reset_index(drop=True)
    w.to_parquet(out)
    return w


def group_weights(groups):
    """Inverse group frequency, normalised to mean 1, clipped at 10 so tiny groups don't dominate."""
    f = groups.map(groups.value_counts(normalize=True))
    w = (1 / f) / (1 / f).mean()
    return w.clip(upper=10).values


# ---------------- metrics ----------------
def _auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def power_mean(x, p=-5):
    x = np.asarray(x, float)
    return np.mean(x ** p) ** (1 / p)


def bias_table(df, p, thr):
    """Per-identity subgroup AUC, BPSN, BNSP, FPR/FNR and FPR gap vs background (Borkan et al. 2019)."""
    y, rows = df.y.values, []
    yhat = p >= thr
    for g in IDS:
        m = df[g].values == 1
        bg_neg, bg_pos = ~m & (y == 0), ~m & (y == 1)
        rows.append(dict(identity=g, n=m.sum(), toxic_rate=y[m].mean(),
                         subgroup_auc=_auc(y[m], p[m]),
                         bpsn_auc=_auc(np.r_[y[m & (y == 0)], y[bg_pos]], np.r_[p[m & (y == 0)], p[bg_pos]]),
                         bnsp_auc=_auc(np.r_[y[m & (y == 1)], y[bg_neg]], np.r_[p[m & (y == 1)], p[bg_neg]]),
                         fpr=yhat[m & (y == 0)].mean(), fnr=(~yhat[m & (y == 1)]).mean(),
                         fpr_gap=yhat[m & (y == 0)].mean() - yhat[bg_neg].mean()))
    return pd.DataFrame(rows)


def summary(df, p, thr):
    """Headline numbers for one model on one split."""
    y, yhat = df.y.values, p >= thr
    bt = bias_table(df, p, thr)
    overall = _auc(y, p)
    tp, fp, fn = (yhat & (y == 1)).sum(), (yhat & (y == 0)).sum(), (~yhat & (y == 1)).sum()
    wg = [((yhat == lab)[(df[g].values == 1) & (y == lab)]).mean() for g in IDS for lab in (0, 1)]
    return dict(auc=overall, f1=2 * tp / (2 * tp + fp + fn), acc=(yhat == y).mean(),
                bias_auc=0.25 * overall + 0.25 * sum(power_mean(bt[c]) for c in ("subgroup_auc", "bpsn_auc", "bnsp_auc")),
                worst_group_acc=min(wg), max_fpr_gap=bt.fpr_gap.max(), mean_abs_fpr_gap=bt.fpr_gap.abs().mean())


def best_f1_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    f1 = [2 * ((p >= t) & (y == 1)).sum() / ((p >= t).sum() + (y == 1).sum()) for t in ts]
    return float(ts[int(np.argmax(f1))])


def ece(y, p, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(y, z):
    """1-D grid search for the temperature that minimises validation NLL of the toxicity logit."""
    ts = np.exp(np.linspace(np.log(0.25), np.log(4), 200))
    nll = [-np.mean(y * np.log(sigmoid(z / t) + 1e-12) + (1 - y) * np.log(1 - sigmoid(z / t) + 1e-12)) for t in ts]
    return float(ts[int(np.argmin(nll))])


def load_scorer(name):
    """name like tfidf_erm or bert_combo -> f(texts) returning raw logits (n, k); column 0 is toxicity."""
    if name.startswith("tfidf"):
        import joblib
        vec, m = joblib.load(MODELS / f"{name}.joblib")
        return lambda texts: m.decision_function(vec.transform(list(texts)))[:, None]
    import bert
    model, tok = bert.load(name.removeprefix("bert_"))
    return lambda texts: bert.score(model, tok, texts)


if __name__ == "__main__":
    # self-checks for the logic that is easy to get subtly wrong
    assert cda_copy("Muslims pray.", np.random.default_rng(0)).split()[0] in {"Christians", "Jews", "Hindus", "Buddhists", "Atheists"}
    v = identity_variants("I am a gay man")
    assert v["straight"] == "I am a straight man" and len(v) == len(SWAP["orientation"])
    assert swap_terms("Gay and gay", lambda w: "straight") == "Straight and straight"
    assert abs(power_mean([0.9, 0.9]) - 0.9) < 1e-9 and power_mean([0.9, 0.5]) < 0.6
    y = np.array([0, 0, 1, 1]); df = pd.DataFrame({"y": y, **{g: [1, 0, 1, 0] for g in IDS}})
    s = summary(df, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert s["auc"] == 1 and s["acc"] == 1 and s["worst_group_acc"] == 1 and s["bias_auc"] == 1
    assert abs(fit_temperature(np.array([0, 1] * 50), np.array([-4.0, 4.0] * 50)) - 0.25) < 1e-6  # perfect separation -> sharpen
    print("common.py self-checks passed")
    d = prepare()
    print(d.split.value_counts().to_dict(), d.group.nunique(), "groups")
    print(d[d.split == "train"].group.value_counts().to_string())

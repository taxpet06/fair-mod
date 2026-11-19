"""All metrics and figures. The only script that reads test predictions.

Per model: temperature fit on val, decision threshold = val-F1-optimal, then on test: AUC/F1/acc, bias metrics,
bootstrap CIs, ECE, risk-coverage, and two counterfactual probes (held-out templates, swapped real comments).
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (prepare, summary, bias_table, best_f1_threshold, fit_temperature, ece, sigmoid, logit,
                    identity_variants, load_scorer, TERM_RE, TERM2CAT, PREDS, RESULTS)

ORDER = ["majority"] + [f"tfidf_{v}" for v in ("erm", "erm80k", "reweight", "cda", "soft", "combo")] \
        + [f"bert_{v}" for v in ("erm", "cda", "reweight", "dro", "soft", "combo")] \
        + [f"bert_distil_{v}" for v in ("erm", "cda", "reweight", "dro", "soft", "combo")]
VARIANTS = ("erm", "cda", "reweight", "dro", "soft", "combo")
FAMILY = {"tfidf": "TF-IDF + LR", "mini": "bert-mini (CPU, 80K)", "distil": "DistilBERT (GPU, full)"}


def family(m):
    return "tfidf" if m.startswith("tfidf") else "distil" if m.startswith("bert_distil") else "mini" if m.startswith("bert") else "other"


def pretty(m):
    v = m.rsplit("_", 1)[1]
    return {"tfidf": "TF-IDF", "mini": "bert-mini", "distil": "DistilBERT"}.get(family(m), m) + (" ERM" if v == "erm" else f" + {v}")

# Held-out CheckList-style suite. Several identities (sikh, queer, arab, mexican, african american) are NOT in the
# CDA swap lexicon, so this also tests whether augmentation generalises beyond the words it saw.
TEMPLATE_IDS = ["muslim", "christian", "jewish", "hindu", "sikh", "atheist", "gay", "straight", "lesbian", "transgender",
                "queer", "black", "white", "asian", "latino", "arab", "mexican", "african american"]
NEUTRAL = ["I am a {} person.", "My neighbor is {} and she is very kind.", "The {} community organized a food drive.",
           "As a {} woman, I disagree with this policy.", "Being {} is part of who I am.", "My best friend is {}.",
           "I'm proud to be {}.", "We had dinner with a {} family last night."]
TOXIC = ["{} people are disgusting.", "I hate {} people.", "All {} people are stupid.", "Get out of here, you {} idiot.",
         "{} people are a plague on this country."]


def load(name, split, d):
    p = pd.read_parquet(PREDS / f"{name}_{split}.parquet")
    return d[d.split == split].merge(p.rename(columns={"toxicity": "z"})[["id", "z"]], on="id")


def risk_coverage(y, p, thr):
    """Sort by distance from the decision threshold (in logit space); error rate among the most confident c%."""
    conf = np.abs(logit(p) - logit(thr))
    err = ((p >= thr) != y)[np.argsort(-conf)]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return np.arange(1, len(err) + 1) / len(err), risk


def bootstrap(df, p, thr, n=50, seed=0):
    rng = np.random.default_rng(seed)
    keys = ("auc", "bias_auc", "worst_group_acc")
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(df), len(df))
        s = summary(df.iloc[i], p[i], thr)
        draws.append([s[k] for k in keys])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return {f"{k}_ci": f"[{a:.4f}, {b:.4f}]" for k, a, b in zip(keys, lo, hi)}


def counterfactual(name, T, thr, d):
    f = load_scorer(name)
    tp = lambda texts: sigmoid(f(texts)[:, 0] / T)
    rows = []
    for kind, temps in (("neutral", NEUTRAL), ("toxic", TOXIC)):
        for t in temps:
            ps = tp([t.format(i) for i in TEMPLATE_IDS])
            rows += [dict(model=name, kind=kind, template=t, identity=i, p=p) for i, p in zip(TEMPLATE_IDS, ps)]
    tdf = pd.DataFrame(rows)
    spread = lambda df: df.groupby("template").p.apply(lambda s: (s - s.mean()).abs().mean()).mean()
    gap = spread(tdf)
    seen = tdf.identity.isin(TERM2CAT)  # identities in the CDA swap lexicon vs held-out ones
    neu = tdf[tdf.kind == "neutral"]
    tox = tdf[tdf.kind == "toxic"]
    # real comments: 2,000 test comments with a lexicon term, every in-category swap, mean |p(swap) - p(original)|
    te = d[d.split == "test"]
    sample = te[te.text.map(lambda t: bool(TERM_RE.search(t)))].sample(2000, random_state=0)
    var = [(k, v) for k, t in enumerate(sample.text) for v in identity_variants(t).values()]
    pv = tp([v for _, v in var])
    po = tp(sample.text)
    real = np.mean([abs(p - po[k]) for (k, _), p in zip(var, pv)])
    stats = dict(cf_template_gap=gap, cf_template_gap_seen=spread(tdf[seen]), cf_template_gap_heldout=spread(tdf[~seen]), cf_real_delta=real, template_neutral_flag_rate=(neu.p >= thr).mean(),
                 template_toxic_recall=(tox.p >= thr).mean(), template_neutral_max_p=neu.p.max())
    return stats, tdf


def figures(S, B, curves, tdf):
    blue, orange, aqua, ink, muted = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    # 1. accuracy vs bias trade-off
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for fam, c in (("tfidf", blue), ("mini", aqua), ("distil", orange)):
        s = S[S.model.map(family) == fam]
        ax.scatter(s.auc, s.bias_auc, s=40, color=c, edgecolor="white", linewidth=1.5, label=FAMILY[fam], zorder=3)
        for _, r in s.iterrows():
            ax.annotate(r.model.rsplit("_", 1)[1], (r.auc, r.bias_auc), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=muted)
    ax.set_xlabel("Overall ROC-AUC (test)"); ax.set_ylabel("Final bias AUC (test)"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_tradeoff.png", dpi=200); plt.close(fig)
    # 2. per-identity FPR, ERM vs best mitigation
    show = list(curves)  # tfidf_erm, the reference ERM, and the best mitigation
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ids = B.identity.unique(); w = 0.8 / len(show)
    for k, (m, c) in enumerate(zip(show, (blue, aqua, orange))):
        v = B[B.model == m].set_index("identity").loc[ids, "fpr"]
        ax.bar(np.arange(len(ids)) + (k - (len(show) - 1) / 2) * w, v, w * 0.9, color=c, label=pretty(m))
    ax.set_xticks(range(len(ids)), ids, rotation=20); ax.set_ylabel("False-positive rate (test)")
    ax.legend(frameon=False, ncol=3, loc="upper left"); fig.tight_layout(); fig.savefig(RESULTS / "fig_fpr_by_identity.png", dpi=200); plt.close(fig)
    # 3. risk-coverage
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for (m, (cov, risk)), c in zip(curves.items(), (blue, aqua, orange)):
        step = max(1, len(cov) // 400)
        ax.plot(cov[::step], risk[::step], color=c, linewidth=2, label=pretty(m))
    ax.set_xlabel("Coverage (share of comments decided automatically)"); ax.set_ylabel("Error rate on decided comments")
    ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(RESULTS / "fig_risk_coverage.png", dpi=200); plt.close(fig)
    # 4. neutral identity templates: mean toxicity probability per identity
    neu = tdf[(tdf.kind == "neutral") & tdf.model.isin(show)].groupby(["identity", "model"]).p.mean().unstack()[show]
    neu = neu.loc[neu[show[0]].sort_values().index]
    fig, ax = plt.subplots(figsize=(6.2, 3.0)); w = 0.8 / len(show)
    for k, (m, c) in enumerate(zip(show, (blue, aqua, orange))):
        ax.bar(np.arange(len(neu)) + (k - (len(show) - 1) / 2) * w, neu[m], w * 0.9, color=c, label=pretty(m))
    ax.set_xticks(range(len(neu)), neu.index, rotation=35, ha="right"); ax.set_ylabel("Mean P(toxic), neutral templates")
    ax.legend(frameon=False, ncol=3, loc="upper left"); fig.tight_layout(); fig.savefig(RESULTS / "fig_templates.png", dpi=200); plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    d = prepare()
    models = [m for m in ORDER if (PREDS / f"{m}_test.parquet").exists()]
    S, B, T, cal, curves = [], [], [], {}, {}
    for m in models:
        va, te = load(m, "val", d), load(m, "test", d)
        temp = 1.0 if m == "majority" else fit_temperature(va.y.values, va.z.values)
        pv, pt = sigmoid(va.z.values / temp), sigmoid(te.z.values / temp)
        thr = 0.5 if m == "majority" else best_f1_threshold(va.y.values, pv)
        cal[m] = dict(temperature=temp, threshold=thr)
        row = dict(model=m, temperature=temp, threshold=thr, **summary(te, pt, thr),
                   ece_raw=ece(te.y.values, sigmoid(te.z.values)), ece_cal=ece(te.y.values, pt))
        vs = summary(va, pv, thr)
        row.update(val_auc=vs["auc"], val_bias_auc=vs["bias_auc"], val_worst_group_acc=vs["worst_group_acc"])
        if m != "majority":
            cov, risk = risk_coverage(te.y.values, pt, thr)
            row.update(aurc=risk.mean(), err_at_80cov=risk[int(0.8 * len(risk)) - 1], **bootstrap(te, pt, thr))
            stats, tdf = counterfactual(m, temp, thr, d)
            row.update(stats); T.append(tdf)
        B.append(bias_table(te, pt, thr).assign(model=m))
        S.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    S, B, T = pd.DataFrame(S), pd.concat(B), pd.concat(T)
    # best mitigation = highest VALIDATION bias AUC within the main transformer family (DistilBERT; bert-mini if absent)
    fam = "distil" if (S.model.map(family) == "distil").any() else "mini"
    cands = S[(S.model.map(family) == fam) & ~S.model.str.endswith("_erm")]
    best = cands.sort_values("val_bias_auc").model.iloc[-1]
    ref = "bert_distil_erm" if fam == "distil" else "bert_erm"
    for m in ("tfidf_erm", ref, best):
        te = load(m, "test", d); curves[m] = risk_coverage(te.y.values, sigmoid(te.z.values / cal[m]["temperature"]), cal[m]["threshold"])
    S.to_csv(RESULTS / "summary.csv", index=False); B.to_csv(RESULTS / "bias_by_identity.csv", index=False)
    T.to_csv(RESULTS / "templates.csv", index=False)
    (RESULTS / "calibration.json").write_text(json.dumps(dict(models=cal, best_by_val_bias_auc=best, erm_reference=ref), indent=1))
    figures(S, B, curves, T)
    print("best mitigated model (chosen on val bias AUC):", best, "vs", ref)
    seeds(d)


def seeds(d=None):
    """DistilBERT seed replicates (0, 1, 2): same protocol per run, then mean and std across seeds."""
    d = prepare() if d is None else d
    rows = []
    for v in VARIANTS:
        for s in (0, 1, 2):
            m = f"bert_distil_{v}" + (f"_s{s}" if s else "")
            if not (PREDS / f"{m}_test.parquet").exists():
                continue
            va, te = load(m, "val", d), load(m, "test", d)
            T = fit_temperature(va.y.values, va.z.values)
            thr = best_f1_threshold(va.y.values, sigmoid(va.z.values / T))
            rows.append(dict(variant=v, seed=s, val_bias_auc=summary(va, sigmoid(va.z.values / T), thr)["bias_auc"],
                             **summary(te, sigmoid(te.z.values / T), thr)))
    R = pd.DataFrame(rows); R.to_csv(RESULTS / "seeds.csv", index=False)
    keys = ["auc", "f1", "bias_auc", "worst_group_acc", "mean_abs_fpr_gap", "max_fpr_gap"]
    A = R.groupby("variant")[keys].agg(["mean", "std", "count"]); A.columns = ["_".join(c) for c in A.columns]
    A.reindex(VARIANTS).to_csv(RESULTS / "seeds_agg.csv")
    print(A.round(4).to_string())


if __name__ == "__main__":
    import sys
    seeds() if sys.argv[1:] == ["seeds"] else main()

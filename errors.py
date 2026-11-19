"""Error analysis on the test split + the curated demo queue.

1. Heuristic tags on every test comment; tag rates among FP / FN / correct for ERM vs the best mitigation.
2. Errors the mitigation fixes vs introduces (same comments, same threshold rule); also for every mitigation of that family.
3. Integrated gradients on the most confident identity-mentioning false positives: is the top word an identity term?
4. A 200-row sheet of errors for manual coding, and the demo queue used by app.py.
5. Data statistics quoted in the report (token lengths, rater fractions, smallest group) -> results/data_stats.csv.
"""
import json, re
import numpy as np, pandas as pd
from common import prepare, sigmoid, IDS, RESULTS, PREDS, MODELS

PROFANITY = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|crap|ass(hole)?s?|bitch\w*|bastard\w*|hell|piss\w*|dick\w*)\b", re.I)
INSULT = re.compile(r"\b(stupid\w*|idiot\w*|moron\w*|dumb\w*|fool\w*|pathetic|ignorant|loser\w*|clown\w*|scum)\b", re.I)
IDWORD = re.compile(r"\b(muslims?|islam\w*|christians?|christianity|catholics?|jews?|jewish|hindus?|atheists?|gays?|lesbians?|"
                    r"homosexuals?|transgender|trans|lgbt\w*|queer|straight|blacks?|whites?|asians?|latinos?|mexicans?|"
                    r"africans?|arabs?|women|woman|men|man|female|male|girls?|boys?|wom[ae]n)\b", re.I)
TAGS = {
    "identity_mention": lambda df: df[IDS].sum(axis=1) > 0,
    "profanity": lambda df: df.text.str.contains(PROFANITY),
    "insult_word": lambda df: df.text.str.contains(INSULT),
    "quote_or_reported": lambda df: df.text.str.contains(r'"|\bsaid\b|\bcalled\b|\bsays\b', regex=True),
    "sarcasm_marker": lambda df: df.text.str.contains(r"/s\b|\blol\b|\byeah,? right\b|\bsure,? \w+\b", case=False, regex=True),
    "negation": lambda df: df.text.str.contains(r"\b(not|no|never|n't)\b", case=False, regex=True),
    "short_lt10w": lambda df: df.text.str.split().str.len() < 10,
    "borderline_label": lambda df: df.toxicity_soft.between(0.3, 0.7),
    "few_raters_lt6": lambda df: df.toxicity_annotator_count < 6,
}


def outcome(y, yhat):
    return np.select([yhat & (y == 0), ~yhat & (y == 1)], ["FP", "FN"], "correct")


def data_stats(d):
    """Numbers the report quotes from the data itself rather than from model outputs."""
    from tokenizers import Tokenizer  # ships with transformers
    tok = Tokenizer.from_file(str(MODELS / "bert_distil_erm" / "tokenizer.json")); tok.no_truncation(); tok.no_padding()
    te, tr = d[d.split == "test"], d[d.split == "train"]
    n = np.array([len(e.ids) for e in tok.encode_batch(te.text.tolist())])  # wordpieces incl. [CLS]/[SEP]
    ben = tr[tr.y == 0]; idm = ben[IDS].sum(axis=1) > 0
    s = dict(test_median_wordpieces=np.median(n), test_share_gt128=(n > 128).mean(), test_share_gt256=(n > 256).mean(),
             test_toxic_share_gt128=(n[te.y.values == 1] > 128).mean(), test_toxic_share=te.y.mean(),
             benign_id_any_rater_toxic=(ben[idm].toxicity_soft > 0).mean(), benign_noid_any_rater_toxic=(ben[~idm].toxicity_soft > 0).mean(),
             benign_id_mean_frac=ben[idm].toxicity_soft.mean(), benign_noid_mean_frac=ben[~idm].toxicity_soft.mean(),
             smallest_train_group=tr.group.value_counts().idxmin(), smallest_train_group_n=tr.group.value_counts().min())
    pd.Series(s, name="value").rename_axis("stat").to_csv(RESULTS / "data_stats.csv")


def fixed_introduced_all(te, cal):
    """Errors each mitigation of the ERM reference's family fixes vs introduces, test split, overall."""
    out = lambda m: outcome(te.y.values, sigmoid(pd.read_parquet(PREDS / f"{m}_test.parquet").set_index("id").loc[te.id, "toxicity"].values
                                                 / cal["models"][m]["temperature"]) >= cal["models"][m]["threshold"])
    ref = cal["erm_reference"]
    e = out(ref)
    rows = []
    for m in [k for k in cal["models"] if k.startswith(ref.rsplit("_", 1)[0] + "_") and k != ref]:
        b = out(m)
        rows += [dict(model=m, error=o, erm_errors=int((e == o).sum()), fixed=int(((e == o) & (b == "correct")).sum()),
                      introduced=int(((b == o) & (e == "correct")).sum()), errors=int((b == o).sum())) for o in ("FP", "FN")]
    pd.DataFrame(rows).to_csv(RESULTS / "errors_fixed_introduced_all.csv", index=False)


def main():
    import warnings; warnings.filterwarnings("ignore", "This pattern")
    cal = json.loads((RESULTS / "calibration.json").read_text())
    best, ref = cal["best_by_val_bias_auc"], cal["erm_reference"]  # best mitigation vs ERM of the same model family
    d = prepare(); te = d[d.split == "test"].copy()
    for t, f in TAGS.items():
        te[t] = f(te).values
    for m in (ref, best):
        z = pd.read_parquet(PREDS / f"{m}_test.parquet").set_index("id").loc[te.id, "toxicity"].values
        c = cal["models"][m]
        te[f"p_{m}"] = sigmoid(z / c["temperature"])
        te[f"out_{m}"] = outcome(te.y.values, te[f"p_{m}"].values >= c["threshold"])
    # 1. tag profile of each outcome
    rows = []
    for m in (ref, best):
        for o in ("FP", "FN", "correct"):
            s = te[te[f"out_{m}"] == o]
            rows.append(dict(model=m, outcome=o, n=len(s), **{t: s[t].mean() for t in TAGS}))
    pd.DataFrame(rows).round(3).to_csv(RESULTS / "errors_tag_profile.csv", index=False)
    # 2. fixed vs introduced, overall and per identity
    wrong_e, wrong_b = te[f"out_{ref}"] != "correct", te[f"out_{best}"] != "correct"
    fx = []
    for name, mask in [("all", np.ones(len(te), bool))] + [(g, te[g].values == 1) for g in IDS]:
        for o in ("FP", "FN"):
            fx.append(dict(subset=name, error=o,
                           erm_errors=int(((te[f"out_{ref}"] == o) & mask).sum()),
                           fixed=int(((te[f"out_{ref}"] == o) & ~wrong_b & mask).sum()),
                           introduced=int(((te[f"out_{best}"] == o) & ~wrong_e & mask).sum()),
                           best_errors=int(((te[f"out_{best}"] == o) & mask).sum())))
    pd.DataFrame(fx).to_csv(RESULTS / "errors_fixed_introduced.csv", index=False)
    fixed_introduced_all(te, cal)
    data_stats(d)
    # 3. IG on the 150 most confident identity-mentioning false positives of each model
    import sys, bert
    if sys.argv[1:] == ["queue"]:
        ig_models = ()
    else:
        ig_models = (ref, best)
    ig_rows = []
    for m in ig_models:
        model, tok = bert.load(m.removeprefix("bert_"))  # DistilBERT IG on CPU: ~1 s per comment
        fp = te[(te[f"out_{m}"] == "FP") & te.identity_mention].nlargest(150, f"p_{m}")
        for _, r in fp.iterrows():
            words, s, _ = bert.integrated_gradients(model, tok, r.text, steps=32)
            top = [words[i] for i in np.argsort(-s)[:3]]
            ig_rows.append(dict(model=m, id=r.id, p=r[f"p_{m}"], top1=top[0], top3=" | ".join(top),
                                top1_identity=bool(IDWORD.fullmatch(top[0])), top3_identity=any(IDWORD.fullmatch(w) for w in top),
                                text=r.text[:300]))
    if ig_rows:
        ig = pd.DataFrame(ig_rows); ig.to_csv(RESULTS / "errors_ig_false_positives.csv", index=False)
        print(ig.groupby("model")[["top1_identity", "top3_identity"]].mean())
    # 4. sheet for manual coding (best model): 100 FP + 100 FN, seeded
    code = pd.concat([te[te[f"out_{best}"] == o].sample(100, random_state=0).assign(error=o) for o in ("FP", "FN")])
    code[["id", "error", f"p_{best}", "toxicity_soft", "toxicity_annotator_count", *TAGS, "text"]].assign(manual_category="") \
        .to_csv(RESULTS / "errors_to_code.csv", index=False)
    # demo queue: curated typed cases + the hardest real ones
    pick = lambda mask, n, case, by=None: (te[mask].nlargest(n, by) if by else te[mask].sample(n, random_state=1)).assign(case=case)
    q = pd.concat([
        pick(te.y.eq(1) & te.toxicity_soft.ge(0.9), 3, "clear toxic"),
        pick(te.y.eq(0) & te.toxicity_soft.eq(0) & ~te.identity_mention, 3, "clear benign"),
        pick(te.toxicity_soft.eq(0) & te.identity_mention & ~te.insult_word & ~te.profanity & (te[f"out_{ref}"] == "FP"), 3,
             "identity mention, 0 raters said toxic, ERM flags it"),
        pick(te.y.eq(0) & te.identity_mention & (te[f"out_{ref}"] == "FP"), 2, "ERM flags it, most raters said non-toxic (label noise?)", f"p_{ref}"),
        pick(te[f"out_{best}"].eq("FN"), 3, "missed toxic (false negative)", "toxicity_soft"),
        pick(te.toxicity_soft.between(0.45, 0.55) & te.toxicity_annotator_count.ge(10), 3, "raters split ~50/50"),
        pick(te.sarcasm_marker & te.y.eq(1), 2, "sarcasm"),
    ])
    typed = pd.DataFrame({"text": ["I am a proud gay man and I love my husband.", "As a Muslim woman, I think this policy is fair.",
                                   "Black lives matter and so does this article.", "Holy shit, this is great news for the town!",
                                   "People like you should go back to where you came from.", "Oh sure, because politicians always tell the truth /s"],
                          "case": ["typed: identity, benign", "typed: identity, benign", "typed: identity, benign",
                                   "typed: profanity, no target", "typed: implicit attack", "typed: sarcasm"]})
    q = pd.concat([typed, q[["text", "case", "y", "toxicity_soft", "toxicity_annotator_count"]]], ignore_index=True)
    q.to_csv(RESULTS / "demo_queue.csv", index=False)
    print(pd.read_csv(RESULTS / "errors_fixed_introduced.csv").head(2).to_string())


if __name__ == "__main__":
    main()

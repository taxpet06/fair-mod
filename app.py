"""Fair-Mod moderator workbench (Gradio). Run: python app.py  ->  http://127.0.0.1:7860

Everything shown is computed live by the selected classifier: score, subtype probabilities, attributions,
identity-swap table, and the flag / allow / defer decision from its calibrated probability.
"""
import json
from functools import lru_cache
import numpy as np, pandas as pd, gradio as gr
from common import prepare, identity_variants, sigmoid, LABELS, MODELS, RESULTS

CAL = json.loads((RESULTS / "calibration.json").read_text()) if (RESULTS / "calibration.json").exists() else {"models": {}}
AVAILABLE = [p.stem for p in sorted(MODELS.glob("tfidf_*.joblib"))] + [p.name for p in sorted(MODELS.glob("bert_*")) if (p / "config.json").exists()]
DEFAULT = CAL.get("best_by_val_bias_auc") if CAL.get("best_by_val_bias_auc") in AVAILABLE else AVAILABLE[-1]
BIAS = pd.read_csv(RESULTS / "bias_by_identity.csv") if (RESULTS / "bias_by_identity.csv").exists() else None
SUMMARY = pd.read_csv(RESULTS / "summary.csv") if (RESULTS / "summary.csv").exists() else None

# queue: curated cases (results/demo_queue.csv, written by errors.py) or a seeded sample of the test split
qpath = RESULTS / "demo_queue.csv"
if qpath.exists():
    QUEUE = pd.read_csv(qpath)
else:  # needs the dataset in data/
    d = prepare()
    QUEUE = d[d.split == "test"].sample(200, random_state=0).assign(case="random test comment")
QUEUE["choice"] = [f"[{c}] {t[:90]}" for c, t in zip(QUEUE.case, QUEUE.text)]


def scorer(name):
    if name.startswith("bert"):
        import bert
        return lambda texts: bert.score(*loaded(name), texts)
    vec, m = loaded(name)
    return lambda texts: m.decision_function(vec.transform(list(texts)))[:, None]


@lru_cache(maxsize=None)
def loaded(name):
    """The raw model objects, loaded once: (vectorizer, LR) for tfidf, (model, tokenizer) for bert."""
    if name.startswith("bert"):
        import bert
        return bert.load(name.removeprefix("bert_"))
    import joblib
    return joblib.load(MODELS / f"{name}.joblib")


def cal(name):
    c = CAL["models"].get(name, {})
    return c.get("temperature", 1.0), c.get("threshold", 0.5)


def decide(p, thr, band):
    if abs(p - thr) < band:
        return "DEFER to human"
    return "FLAG" if p >= thr else "ALLOW"


def attributions(name, text):
    """Word -> contribution to the toxicity score. bert: integrated gradients. tfidf: coefficient x tf-idf weight (unigrams)."""
    if name.startswith("bert"):
        import bert
        words, s, _ = bert.integrated_gradients(*loaded(name), text)
    else:
        vec, m = loaded(name)
        x = vec.transform([text]); vocab = vec.vocabulary_
        words = vec.build_analyzer()(text)
        words = [w for w in words if " " not in w]  # unigrams only; bigram weight is not shown
        s = np.array([m.coef_[0][vocab[w]] * x[0, vocab[w]] if w in vocab else 0.0 for w in words])
    scale = np.abs(s).max() or 1.0
    return [(w + " ", float(v / scale)) for w, v in zip(words, s)]


def run(text, name, band):
    if not text or not text.strip():
        raise gr.Error("Type or pick a comment first.")
    T, thr = cal(name)
    f = scorer(name)
    variants = {k: v for k, v in identity_variants(text).items() if v != text}
    texts = [text] + list(variants.values())
    z = f(texts)
    p = sigmoid(z[:, 0] / T)
    dec = decide(p[0], thr, band)
    md = (f"## {dec}\n\nCalibrated P(toxic) = **{p[0]:.3f}** · decision threshold {thr:.2f} · defer band ±{band:.2f} · "
          f"temperature {T:.2f} · model `{name}`")
    gold = QUEUE[QUEUE.text == text]
    if len(gold) and pd.notna(gold.iloc[0].toxicity_soft):  # typed demo cases have no gold label
        r = gold.iloc[0]
        md += (f"\n\n**Annotators:** {r.toxicity_soft:.0%} of {int(r.toxicity_annotator_count)} raters called this toxic "
               f"(gold label: {'toxic' if r.y else 'non-toxic'}).")
    subtypes = {l: float(sigmoid(z[0, k])) for k, l in enumerate(LABELS) if k} if z.shape[1] > 1 else None
    if subtypes is None:
        md += "\n\n_Subtype probabilities come from the bert models only._"
    if variants:
        swap = pd.DataFrame({"identity term": ["(original)"] + list(variants), "P(toxic)": p.astype(float).round(3),
                             "change vs original": (p - p[0]).astype(float).round(3), "decision": [decide(q, thr, band) for q in p]})
    else:
        swap = pd.DataFrame({"identity term": ["no identity term from the swap lexicon found"]})
    return md, subtypes, attributions(name, text), swap


def pick(choice):
    return QUEUE.loc[QUEUE.choice == choice, "text"].iloc[0] if choice else ""


def group_table(name):
    if BIAS is None:
        return pd.DataFrame({"note": ["run evaluate.py first"]}), pd.DataFrame()
    b = BIAS[BIAS.model == name].drop(columns="model").round(3)
    s = SUMMARY[SUMMARY.model == name][["auc", "f1", "acc", "bias_auc", "worst_group_acc", "ece_cal", "cf_template_gap", "cf_real_delta"]].round(3)
    return b, s


with gr.Blocks(title="Fair-Mod") as demo:
    gr.Markdown("# Fair-Mod: bias-aware toxicity moderation\nDecision support for a human moderator, not an automatic removal "
                "system. **Content warning:** examples contain offensive language from real news comments.")
    with gr.Tab("Moderate"):
        with gr.Row():
            with gr.Column(scale=3):
                choice = gr.Dropdown(QUEUE.choice.tolist(), value=None, label="Pick a comment from the queue (real test comments + curated cases)")
                text = gr.Textbox(lines=4, label="Comment", placeholder="...or type your own comment here")
            with gr.Column(scale=2):
                model = gr.Dropdown(AVAILABLE, value=DEFAULT, label="Classifier")
                band = gr.Slider(0, 0.4, value=0.15, step=0.01, label="Defer band: send to a human when |P(toxic) - threshold| is below this")
                go = gr.Button("Score comment", variant="primary")
        decision = gr.Markdown()
        with gr.Row():
            sub = gr.Label(label="Subtype probabilities (uncalibrated)", num_top_classes=6)
            attr = gr.HighlightedText(label="Which words pushed the score (red = toward toxic, blue = away)", combine_adjacent=False, show_legend=False)
        swap = gr.Dataframe(label="Identity swap: same comment, only the identity term changed")
        choice.change(pick, choice, text)
        go.click(run, [text, model, band], [decision, sub, attr, swap])
        text.submit(run, [text, model, band], [decision, sub, attr, swap])
    with gr.Tab("Group metrics"):
        gm = gr.Dropdown(AVAILABLE, value=DEFAULT, label="Classifier")
        summ = gr.Dataframe(label="Test-set summary")
        tab = gr.Dataframe(label="Per-identity test metrics (threshold tuned on validation)")
        gm.change(group_table, gm, [tab, summ]); demo.load(group_table, gm, [tab, summ])
        figs = [str(RESULTS / f) for f in ("fig_tradeoff.png", "fig_fpr_by_identity.png", "fig_templates.png", "fig_risk_coverage.png") if (RESULTS / f).exists()]
        if figs:
            gr.Gallery(figs, columns=2, height=480, object_fit="contain", label="Figures (click to enlarge)")
    with gr.Tab("Limitations"):
        gr.Markdown((RESULTS.parent / "LIMITATIONS.md").read_text() if (RESULTS.parent / "LIMITATIONS.md").exists() else "")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

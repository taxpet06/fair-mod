"""Generate LaTeX tables + inline-number macros from results/ so the report cannot drift from the code."""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import prepare, RESULTS

OUT = Path(__file__).parent
S = pd.read_csv(RESULTS / "summary.csv").set_index("model")
B = pd.read_csv(RESULTS / "bias_by_identity.csv")
CAL = json.loads((RESULTS / "calibration.json").read_text())
BEST, REF = CAL["best_by_val_bias_auc"], CAL["erm_reference"]
NAME = {"majority": "B1 majority", "tfidf_erm": "B2 ERM", "tfidf_erm80k": "\\quad ERM, 80K subsample", "tfidf_reweight": "\\quad + reweight",
        "tfidf_cda": "\\quad + CDA", "tfidf_soft": "\\quad + soft labels", "tfidf_combo": "\\quad + combo",
        "bert_erm": "bert-mini ERM (80K, CPU)", "bert_reweight": "\\quad + reweight", "bert_cda": "\\quad + CDA", "bert_dro": "\\quad + group DRO",
        "bert_soft": "\\quad + soft labels", "bert_combo": "\\quad + combo",
        "bert_distil_erm": "B3 DistilBERT ERM", "bert_distil_reweight": "\\quad + reweight", "bert_distil_cda": "\\quad + CDA",
        "bert_distil_dro": "\\quad + group DRO", "bert_distil_soft": "\\quad + soft labels", "bert_distil_combo": "\\quad + combo"}
ERMS = ("tfidf_erm", "bert_erm", "bert_distil_erm")


def fam(m):
    return "tfidf" if m.startswith("tfidf") else "distil" if m.startswith("bert_distil") else "mini" if m.startswith("bert") else "other"
f3 = lambda x: f"{x:.3f}"


def ci(s):  # "[0.91, 0.92]" -> "\pm half-width"
    lo, hi = map(float, s.strip("[]").split(","))
    return f"{{\\scriptsize$\\pm${(hi - lo) / 2:.3f}}}"


def bold_best(col, rows, higher=True):
    v = S.loc[rows, col]
    return v.idxmax() if higher else v.idxmin()


def main_table():
    rows = [m for m in NAME if m in S.index]
    cols = [("auc", True), ("f1", True), ("bias_auc", True), ("worst_group_acc", True), ("mean_abs_fpr_gap", False), ("ece_cal", False)]
    best = {(c, f): bold_best(c, [m for m in rows if fam(m) == f], h) for c, h in cols for f in ("tfidf", "mini", "distil")
            if any(fam(m) == f for m in rows)}
    out = ["\\begin{table*}[t]\\centering\\small", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lcccccc}\\toprule",
           "Model & ROC-AUC $\\uparrow$ & F1 $\\uparrow$ & Bias AUC $\\uparrow$ & Worst-group acc $\\uparrow$ & Mean $|$FPR gap$|$ $\\downarrow$ & ECE $\\downarrow$\\\\\\midrule"]
    for m in rows:
        r = S.loc[m]
        cells = []
        for c, _ in cols:
            v = f3(r[c])
            if best.get((c, fam(m))) == m:
                v = f"\\textbf{{{v}}}"
            if c in ("auc", "bias_auc", "worst_group_acc") and isinstance(r.get(c + "_ci"), str):
                v += " " + ci(r[c + "_ci"])
            cells.append(v)
        if m in ERMS:
            out.append("\\midrule")
        out.append(f"{NAME[m]} & " + " & ".join(cells) + "\\\\")
    out += ["\\bottomrule\\end{tabular}}",
            "\\caption{Test results (WILDS test, " + f"{NTEST:,}" + " comments). Thresholds and temperatures are fit on validation; "
            "$\\pm$ is half the 95\\% bootstrap interval (50 resamples of the test set; it does not include training-seed variance). "
            "Bias AUC is the \\citet{borkan2019nuanced} final metric (generalised mean, $p{=}{-}5$, of subgroup, BPSN and BNSP AUC, averaged with overall AUC). "
            "Worst-group accuracy is the minimum over the 16 identity$\\times$label groups. FPR gap is identity-group FPR minus background FPR. "
            "Bold marks the best value within each model family.}\\label{tab:main}", "\\end{table*}"]
    (OUT / "tab_main.tex").write_text("\n".join(out))


def cf_table():
    rows = [m for m in NAME if m in S.index and m != "majority"]
    out = ["\\begin{table}[t]\\centering\\small\\setlength\\tabcolsep{3pt}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}\\toprule",
           "& \\multicolumn{3}{c}{Template spread $\\downarrow$} & Real & Neutral\\\\",
           "Model & all & seen & held-out & $|\\Delta p|$ $\\downarrow$ & flagged $\\downarrow$\\\\\\midrule"]
    for m in rows:
        r = S.loc[m]
        if m in ERMS[1:]:
            out.append("\\midrule")
        out.append(f"{NAME[m]} & " + " & ".join(f3(r[c]) for c in ("cf_template_gap", "cf_template_gap_seen", "cf_template_gap_heldout", "cf_real_delta")) +
                   f" & {r.template_neutral_flag_rate:.0%}".replace("%", "\\%") + "\\\\")
    out += ["\\bottomrule\\end{tabular}}",
            "\\caption{Counterfactual probes. \\emph{Template spread}: mean absolute deviation of $P(\\text{toxic})$ from the template mean across 18 identity terms "
            "(8 neutral and 5 toxic templates), split into terms in the CDA lexicon (\\emph{seen}) and 5 held-out terms (sikh, queer, arab, mexican, african american). "
            "\\emph{Real}: mean $|p(\\text{swap})-p(\\text{original})|$ over in-category swaps of 2{,}000 test comments. "
            "\\emph{Neutral flagged}: share of the 144 neutral template sentences above the model's threshold.}\\label{tab:cf}", "\\end{table}"]
    (OUT / "tab_cf.tex").write_text("\n".join(out))


def identity_table():
    show = ["tfidf_erm", REF, BEST]
    piv = B[B.model.isin(show)].pivot(index="identity", columns="model")
    out = ["\\begin{table*}[t]\\centering\\small", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrr" + "ccc" * 3 + "}\\toprule",
           "& & & \\multicolumn{3}{c}{BPSN AUC $\\uparrow$} & \\multicolumn{3}{c}{FPR $\\downarrow$} & \\multicolumn{3}{c}{FNR $\\downarrow$}\\\\",
           "Identity & $n$ & toxic & " + " & ".join(["TF-IDF", "ERM", "best"] * 3) + "\\\\\\midrule"]
    for g in piv.index:
        r = piv.loc[g]
        out.append(f"{g.replace('_', ' ')} & {int(r[('n', REF)]):,} & {r[('toxic_rate', REF)]:.0%} & ".replace("%", "\\%") +
                   " & ".join(f3(r[(c, m)]) for c in ("bpsn_auc", "fpr", "fnr") for m in show) + "\\\\")
    out += ["\\bottomrule\\end{tabular}}", f"\\caption{{Per-identity test metrics for TF-IDF ERM, DistilBERT ERM, and the best mitigated DistilBERT model by validation bias AUC "
            f"(\\texttt{{{BEST.replace('_', '\\_')}}}). BPSN AUC is low when non-toxic comments that mention the group score above toxic comments that don't.}}\\label{{tab:identity}}", "\\end{table*}"]
    (OUT / "tab_identity.tex").write_text("\n".join(out))


def error_tables():
    t = pd.read_csv(RESULTS / "errors_tag_profile.csv")
    tags = ["identity_mention", "profanity", "insult_word", "quote_or_reported", "sarcasm_marker", "borderline_label", "few_raters_lt6"]
    out = ["\\begin{table}[t]\\centering\\small\\setlength\\tabcolsep{3pt}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "c" * 6 + "}\\toprule",
           "& \\multicolumn{3}{c}{bert ERM} & \\multicolumn{3}{c}{best}\\\\", "Tag & FP & FN & ok & FP & FN & ok\\\\\\midrule"]
    for g in tags:
        vals = [t[(t.model == m) & (t.outcome == o)][g].iloc[0] for m in (REF, BEST) for o in ("FP", "FN", "correct")]
        out.append(g.replace("_", " ").replace("lt6", "($<$6)").replace("lt10w", "($<$10 words)") + " & " + " & ".join(f"{v:.0%}".replace("%", "\\%") for v in vals) + "\\\\")
    n = [int(t[(t.model == m) & (t.outcome == o)].n.iloc[0]) for m in (REF, BEST) for o in ("FP", "FN", "correct")]
    out.append("$n$ & " + " & ".join(f"{x:,}" for x in n) + "\\\\")
    out += ["\\bottomrule\\end{tabular}}", "\\caption{Share of test comments carrying each heuristic tag, by outcome. \\emph{borderline label}: 30--70\\% of raters said toxic.}\\label{tab:tags}", "\\end{table}"]
    (OUT / "tab_tags.tex").write_text("\n".join(out))
    f = pd.read_csv(RESULTS / "errors_fixed_introduced_all.csv").set_index(["model", "error"])
    out = ["\\begin{table}[t]\\centering\\small\\setlength\\tabcolsep{3pt}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrrrrr}\\toprule",
           "& \\multicolumn{3}{c}{False positives} & \\multicolumn{3}{c}{False negatives}\\\\",
           "DistilBERT & fixed & new & total & fixed & new & total\\\\\\midrule",
           f"ERM & & & {f.erm_errors.iloc[0]:,} & & & {f.erm_errors.iloc[1]:,}\\\\"]
    for m in dict.fromkeys(f.index.get_level_values(0)):
        out.append(NAME[m].replace("\\quad ", "") + " & " + " & ".join(f"{f.loc[(m, e), c]:,}" for e in ("FP", "FN") for c in ("fixed", "introduced", "errors")) + "\\\\")
    out += ["\\bottomrule\\end{tabular}}", "\\caption{Test errors of each DistilBERT mitigation against ERM (seed 0, each model at its own validation "
            "threshold). \\emph{fixed}: ERM errors the mitigation gets right; \\emph{new}: errors ERM did not make.}\\label{tab:fixed}", "\\end{table}"]
    (OUT / "tab_fixed.tex").write_text("\n".join(out))


def hand_coded():
    """Author's hand labels (results/errors_hand_coded.csv) for the 200 sampled errors in results/errors_to_code.csv."""
    h = pd.read_csv(RESULTS / "errors_hand_coded.csv", encoding="utf-8-sig")  # BOM: exported from a spreadsheet
    d = pd.read_csv(RESULTS / "errors_to_code.csv").merge(h, on="id", suffixes=("", "_h"), validate="1:1")
    assert len(d) == 200 and (d.error == d.error_h).all() and d.human_label.notna().all()
    ct = pd.crosstab(d.human_label, d.error)[["FP", "FN"]]
    ct = ct.assign(n=ct.FP + ct.FN).sort_values(["n", "FP"], ascending=False)
    show = {"Other (explain in Notes)": "Other: disguised insult or truncation"}
    out = ["\\begin{table}[t]\\centering\\small", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrr}\\toprule", "Reason (hand-coded) & FP & FN & Total\\\\\\midrule"]
    out += [f"{show.get(k, k)} & {r.FP} & {r.FN} & {r.n}\\\\" for k, r in ct.iterrows()]
    out += [f"Total & {ct.FP.sum()} & {ct.FN.sum()} & {ct.n.sum()}\\\\", "\\bottomrule\\end{tabular}}",
            "\\caption{Main reason for each of 200 test errors of the best model (100 false positives and 100 false negatives, "
            "sampled at random), labelled by hand by the author. One label per error.}\\label{tab:hand}", "\\end{table}"]
    (OUT / "tab_hand.tex").write_text("\n".join(out))
    of = lambda m, k: f"{int(m.sum())} of {int(k.sum())}"
    lab = d.human_label
    return {"HandFPIdentity": ct.loc["Identity mention, not hostile", "FP"], "HandFPCivil": ct.loc["Harsh but civil disagreement", "FP"],
            "HandFNCivil": ct.loc["Harsh but civil disagreement", "FN"], "HandFNImplicit": ct.loc["Implicit or coded hostility", "FN"],
            "HandFNSarcasm": ct.loc["Sarcasm or irony", "FN"], "HandOther": ct.loc["Other (explain in Notes)", "n"],
            "HandGoldWrong": ct.loc["Gold label looks wrong", "n"], "HandAmbiguous": ct.loc["Genuinely ambiguous", "n"],
            "HandIdTagFP": of(d.identity_mention[d.error == "FP"], d.error == "FP"),
            "HandQuoteTag": of(d.quote_or_reported[lab == "Quoting or reporting toxicity"], lab == "Quoting or reporting toxicity"),
            "HandQuoteTagElse": of(d.quote_or_reported[lab != "Quoting or reporting toxicity"], lab != "Quoting or reporting toxicity"),
            "HandSarcTag": of(d.sarcasm_marker[lab == "Sarcasm or irony"], lab == "Sarcasm or irony")}


def data_macros():
    ds = pd.read_csv(RESULTS / "data_stats.csv").set_index("stat").value
    pct = lambda k, nd=0: f"{float(ds[k]):.{nd}%}".replace("%", "\\%")
    return {"TokMedian": f"{float(ds['test_median_wordpieces']):.0f}", "TokOverOneTwoEight": pct("test_share_gt128"),
            "TokOverTwoFiveSix": pct("test_share_gt256", 1), "TokToxicOverOneTwoEight": pct("test_toxic_share_gt128"),
            "ToxicShare": pct("test_toxic_share"), "BenignIdRater": pct("benign_id_any_rater_toxic"),
            "BenignNoIdRater": pct("benign_noid_any_rater_toxic"), "BenignIdFrac": f"{float(ds['benign_id_mean_frac']):.3f}",
            "BenignNoIdFrac": f"{float(ds['benign_noid_mean_frac']):.3f}", "SmallestGroupN": f"{int(ds['smallest_train_group_n']):,}"}


def numbers():
    global NTEST
    d = prepare()
    raw = {"train": 267516, "val": 45047, "test": 132730}
    n = d.split.value_counts()
    te = d[d.split == "test"]
    NTEST = int(n["test"])
    ig = pd.read_csv(RESULTS / "errors_ig_false_positives.csv").groupby("model")[["top1_identity", "top3_identity"]].mean()
    fx = pd.read_csv(RESULTS / "errors_fixed_introduced.csv")
    a = fx[fx.subset == "all"].set_index("error")
    m = {"MedianRaters": f"{te.toxicity_annotator_count.median():.0f}", "BorderlineShare": f"{te.toxicity_soft.between(.3, .7).mean():.0%}".replace("%", "\\%"),
         "LeakTrain": raw["train"] - n["train"], "LeakVal": raw["val"] - n["val"], "LeakTest": raw["test"] - n["test"],
         "NTrain": f"{n['train']:,}", "NVal": f"{n['val']:,}", "NTest": f"{n['test']:,}",
         "Best": f"\\texttt{{{BEST.replace('_', '\\_')}}}",
         "IGTopErm": f"{ig.loc[REF, 'top1_identity']:.0%}".replace("%", "\\%"), "IGTopBest": f"{ig.loc[BEST, 'top1_identity']:.0%}".replace("%", "\\%"),
         "IGTopThreeErm": f"{ig.loc[REF, 'top3_identity']:.0%}".replace("%", "\\%"), "IGTopThreeBest": f"{ig.loc[BEST, 'top3_identity']:.0%}".replace("%", "\\%"),
         "FPFixed": f"{a.loc['FP', 'fixed']:,}", "FPIntro": f"{a.loc['FP', 'introduced']:,}", "FPErm": f"{a.loc['FP', 'erm_errors']:,}", "FPBest": f"{a.loc['FP', 'best_errors']:,}",
         "FNFixed": f"{a.loc['FN', 'fixed']:,}", "FNIntro": f"{a.loc['FN', 'introduced']:,}", "FNErm": f"{a.loc['FN', 'erm_errors']:,}", "FNBest": f"{a.loc['FN', 'best_errors']:,}",
         **hand_coded(), **data_macros()}
    for model in S.index:  # \auc{bert_erm} etc. would need a key-value package; plain macros per model+metric instead
        key = "".join(w.capitalize() for w in model.replace("80k", "Eighty").split("_"))
        for c in ("auc", "f1", "acc", "bias_auc", "worst_group_acc", "mean_abs_fpr_gap", "max_fpr_gap", "ece_raw", "ece_cal", "cf_template_gap",
                  "cf_template_gap_seen", "cf_template_gap_heldout", "cf_real_delta", "template_neutral_flag_rate", "aurc", "err_at_80cov", "threshold", "temperature"):
            if c in S.columns and pd.notna(S.loc[model, c]):
                m[key + "".join(w.capitalize() for w in c.replace("80", "eighty").split("_"))] = f3(S.loc[model, c])
    words = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
    clean = lambda k: "".join(words.get(ch, ch) for ch in k)  # LaTeX macro names cannot contain digits
    (OUT / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{clean(k)}}}{{{v}}}" for k, v in m.items()) + "\n")


def seed_table():
    A = pd.read_csv(RESULTS / "seeds_agg.csv").set_index("variant")
    cols = ["auc", "f1", "bias_auc", "worst_group_acc", "mean_abs_fpr_gap"]
    out = ["\\begin{table}[t]\\centering\\small\\setlength\\tabcolsep{2.5pt}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lccccc}\\toprule",
           "DistilBERT & AUC & F1 & Bias AUC & Worst-gr. & $|$FPR gap$|$\\\\\\midrule"]
    for v, r in A.iterrows():
        cells = [f"{r[c + '_mean']:.3f}" + (f"{{\\scriptsize$\\pm${r[c + '_std']:.3f}}}" if r[c + "_count"] > 1 else "") for c in cols]
        out.append(f"{v} ({int(r.auc_count)}) & " + " & ".join(cells) + "\\\\")
    out += ["\\bottomrule\\end{tabular}}", "\\caption{DistilBERT across training seeds: mean $\\pm$ standard deviation (number of seeds in parentheses). "
            "Each seed gets its own validation temperature and threshold.}\\label{tab:seeds}", "\\end{table}"]
    (OUT / "tab_seeds.tex").write_text("\n".join(out))


if __name__ == "__main__":
    numbers(); main_table(); cf_table(); identity_table(); error_tables(); seed_table()
    print("wrote numbers.tex and 7 tables")

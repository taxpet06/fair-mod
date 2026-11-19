# Fair-Mod: bias-aware toxicity moderation

Fair-Mod is a moderator workbench for English news comments. It scores toxicity, shows which words drove the score,
re-scores the comment with identity terms swapped, and defers uncertain cases to a human. Underneath it is a
controlled comparison of four bias mitigations (counterfactual augmentation, group reweighting, group DRO, soft labels)
against plain training. The comparison runs for a TF-IDF + logistic regression model and for two fine-tuned transformers (DistilBERT on a GPU cluster, bert-mini on a laptop CPU),
and every model is evaluated with identity-subgroup metrics on CivilComments-WILDS.

Course project for COSC 72. The written report is submitted separately.

| Deliverable | Where |
|---|---|
| Demo video (99 s) | [`demo/fairmod_demo.mp4`](demo/fairmod_demo.mp4) |
| Slides (5) | [`slides/slides.pdf`](slides/slides.pdf); `slides/slides.html` is the presenter version (arrow keys, `N` for notes) |
| Results behind every number | [`results/`](results/) |

> Content warning: the data and demo contain offensive language.

**Dataset:** [CivilComments-WILDS on Hugging Face](https://huggingface.co/datasets/pietrolesci/civilcomments-wilds)
(CC0; mirror of the [WILDS](https://wilds.stanford.edu/datasets/) CivilComments benchmark). The data is not stored in
this repository because of its size; `download_data.sh` fetches it.

## Setup

Python 3.12. Everything except the DistilBERT runs works on a CPU (12-core laptop); DistilBERT was trained on a GPU
cluster (see "GPU runs").

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_data.sh            # CivilComments-WILDS (CC0), ~240 MB, into data/wilds/
```

## Reproduce

| Step | Command | Time on the laptop |
|---|---|---|
| Data prep + self-checks | `python common.py` | 10 s |
| B1 majority + TF-IDF variants | `python tfidf.py` | ~10 min |
| bert-mini variants (CPU) | `python bert.py erm cda reweight dro soft combo` | ~3 h |
| DistilBERT variants x 3 seeds (GPU) | see "GPU runs" below | 2-25 min per run |
| IG completeness check | `python bert.py selfcheck` | 20 s |
| All metrics, CIs, counterfactual probes, figures | `python evaluate.py` | ~25 min |
| Error analysis + demo queue | `python errors.py` | ~5 min |
| Report tables + inline numbers (reads `results/`, incl. the hand-coded errors) | `python report/build_tables.py` | ~10 s |
| Demo app | `python app.py`, then open http://127.0.0.1:7860 (needs the data and at least one trained model from the steps above) | |

Trained models (~1.9 GB) and per-comment predictions are not stored in the repository; the steps above recreate
them. Seeds are fixed (`random_state=0`, `torch.manual_seed(0)`), but CPU kernel nondeterminism can move bert numbers in
the third decimal place.

## GPU runs (Dartmouth Discovery)

`bert.py` reads `BASE`, `TOKENIZER`, `TAG`, `SEED`, `LR`, `N_TRAIN` from the environment and uses CUDA when it is
available. The DistilBERT results come from:

```bash
BASE=distilbert/distilbert-base-uncased TOKENIZER=distilbert/distilbert-base-uncased TAG=distil LR=5e-5 N_TRAIN=0 SEED=0 python bert.py erm
```

On Discovery, `cluster/driver.sh` runs every (variant, seed) as its own `srun` job on `gpu_preempt` under the `free`
account. GPU nodes cannot read home or `/jumbo`, so `cluster/job.sh` receives the code and `data/prepared.parquet`
as a tar on stdin, builds a venv in the node's `/tmp`, trains, and writes `models/` and `preds/` back as a tar on
stdout. V100 nodes (no kernel image in the torch 2.15 wheel) and the small-`/tmp` turing nodes are excluded. Copy
`out/models` and `out/preds` into this folder, then run `evaluate.py`.

## Files

| File | What it does |
|---|---|
| `common.py` | data prep (normalisation, dedup, leakage filter, soft-label join, groups), identity swap lexicon, all metrics, temperature scaling |
| `tfidf.py` | majority baseline and TF-IDF + LR variants |
| `bert.py` | transformer training for each variant (bert-mini on CPU, DistilBERT on GPU), batched scoring, integrated gradients |
| `cluster/` | Slurm job + driver for the GPU runs |
| `evaluate.py` | the only script that reads test predictions: summary metrics, bias tables, bootstrap CIs, ECE, risk-coverage, template + real-comment counterfactual probes, figures |
| `errors.py` | error tags, fixed vs introduced errors, IG on false positives, manual-coding sheet, demo queue |
| `coding_sheet.py` | builds the spreadsheet for hand-coding the 200 sampled errors; the author's hand labels are in `results/errors_hand_coded.csv` |
| `app.py` | Gradio moderator workbench |
| `results/` | the CSVs behind the report and slides (`report/build_tables.py` turns them into tables and inline numbers); `errors_hand_coded.csv` holds the author's hand labels |
| `LIMITATIONS.md` | known limitations, also shown inside the app |

## Data

The project uses CivilComments-WILDS (Koh et al., 2021), a subset of Civil Comments (Borkan et al., 2019) with
identity annotations. It is CC0 and comes from the [`pietrolesci/civilcomments-wilds`](https://huggingface.co/datasets/pietrolesci/civilcomments-wilds) mirror on Hugging Face. The
standard train/val/test split is used. Soft labels (annotator fractions) are joined from the mirror's `raw/` file by
comment `id`. `prepare()` drops train or val comments whose normalised text also appears in test, and train comments
whose text appears in val. It also removes duplicates within each split.

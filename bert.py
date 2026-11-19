"""Transformer toxicity model (bert-mini on CPU by default; any HF encoder via BASE/TOKENIZER on GPU): training (python bert.py VARIANT), batched scoring, integrated gradients.

Variants: erm | cda | reweight | dro | soft | combo (= cda + reweight + soft).
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from common import prepare, cda_copy, group_weights, TERM_RE, LABELS, MODELS, PREDS

ENV = os.environ.get
BASE = ENV("BASE", "prajjwal1/bert-mini")
TOKENIZER = ENV("TOKENIZER", "google-bert/bert-base-uncased")  # prajjwal1 repos ship no tokenizer; they use the uncased BERT vocab
TAG, SEED = ENV("TAG", ""), int(ENV("SEED", 0))  # TAG names the model family (e.g. distil); SEED > 0 adds a _sN suffix
MAXLEN, BATCH, LR, DRO_ETA = 128, int(ENV("BATCH", 32)), float(ENV("LR", 1e-4)), 0.01
N_TRAIN = int(ENV("N_TRAIN", 80_000))  # 0 = the full training split
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("SMOKE")  # tiny run that checks the whole train -> save -> predict path
torch.set_num_threads(int(os.environ.get("THREADS", 12)))


def run_name(variant):
    return "bert_" + (TAG + "_" if TAG else "") + variant + (f"_s{SEED}" if SEED else "")


def base_model():  # prajjwal1 configs lack model_type, so AutoModel can't read them
    cls = BertForSequenceClassification if "prajjwal1" in BASE else AutoModelForSequenceClassification
    return cls.from_pretrained(BASE, num_labels=len(LABELS))


def load(variant):
    """variant is the part after 'bert_', e.g. 'combo' or 'distil_erm'."""
    path = MODELS / f"bert_{variant}"
    return AutoModelForSequenceClassification.from_pretrained(path).to(DEV).eval(), AutoTokenizer.from_pretrained(path)


def batches(texts, tok, bs, order):
    """Yield (index_array, encoded batch) with dynamic padding."""
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        yield idx, tok([texts[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")


@torch.no_grad()
def score(model, tok, texts, bs=128):
    """Raw logits (n, 7) for a list of texts. Sorting by length makes CPU inference ~2x faster."""
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), len(LABELS)), np.float32)
    for idx, enc in batches(texts, tok, bs, order):
        out[idx] = model(**enc.to(model.device)).logits.float().cpu().numpy()
    return out


def integrated_gradients(model, tok, text, steps=48):
    """Word-level IG attributions for the toxicity logit; baseline = [PAD] embeddings for every non-special token."""
    enc = tok(text, truncation=True, max_length=MAXLEN, return_tensors="pt").to(model.device)
    ids = enc.input_ids
    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        x = emb_layer(ids)
        special = (ids == tok.cls_token_id) | (ids == tok.sep_token_id)
        base = torch.where(special[..., None], x, emb_layer(torch.full_like(ids, tok.pad_token_id)))
    a = torch.linspace(0, 1, steps, device=model.device)[:, None, None]
    path = (base + a * (x - base)).requires_grad_(True)
    logit = model(inputs_embeds=path, attention_mask=enc.attention_mask.expand(steps, -1)).logits[:, 0]
    (g,) = torch.autograd.grad(logit.sum(), path)
    attr = ((x - base)[0] * g.mean(0)).sum(-1).detach().cpu().numpy()
    toks = tok.convert_ids_to_tokens(ids[0].cpu())
    words, scores = [], []
    for t, s in zip(toks[1:-1], attr[1:-1]):  # merge wordpieces back into words
        if t.startswith("##") and words:
            words[-1] += t[2:]; scores[-1] += s
        else:
            words.append(t); scores.append(s)
    return words, np.array(scores), float((logit[-1] - logit[0]).detach())  # last value: f(x) - f(baseline), for the completeness check


def train(variant):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    d = prepare()
    tr = d[d.split == "train"]
    tr = (tr.sample(N_TRAIN, random_state=0) if N_TRAIN else tr).reset_index(drop=True)  # same subsample for every seed
    use_cda, use_rw, use_soft = variant in ("cda", "combo"), variant in ("reweight", "combo"), variant in ("soft", "combo")
    if use_cda:
        has = tr[tr.text.map(lambda t: bool(TERM_RE.search(t)))]
        tr = pd.concat([tr, has.assign(text=[cda_copy(t, rng) for t in has.text])], ignore_index=True)
        print(f"CDA added {len(has)} counterfactual copies")
    soft = tr[[l + "_soft" for l in LABELS]].values.astype(np.float32)
    targets = torch.tensor(soft if use_soft else (soft >= 0.5).astype(np.float32))
    weights = torch.tensor(group_weights(tr.group) if use_rw else np.ones(len(tr)), dtype=torch.float32)
    gidx = torch.tensor(pd.factorize(tr.group)[0]).to(DEV)
    q = torch.ones(int(gidx.max()) + 1, device=DEV) / (int(gidx.max()) + 1)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = base_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # bucket by length inside shuffled blocks: dynamic padding without a fully sorted (non-random) epoch
    perm = rng.permutation(len(tr))
    lens = tr.text.str.len().values
    blocks = [perm[i:i + BATCH * 50] for i in range(0, len(perm), BATCH * 50)]
    order = np.concatenate([b[np.argsort(lens[b])] for b in blocks])
    bidx = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    rng.shuffle(bidx)
    order = np.concatenate(bidx)
    steps = len(bidx)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train(); t0 = time.time(); texts = tr.text.tolist()
    for step, (idx, enc) in enumerate(batches(texts, tok, BATCH, order)):
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = model(**enc.to(DEV)).logits
        per = F.binary_cross_entropy_with_logits(logits.float(), targets[idx].to(DEV), reduction="none").mean(1)
        if variant == "dro":  # online group DRO (Sagawa et al. 2020): upweight groups with high loss
            g = gidx[idx]
            gl = torch.zeros_like(q).index_add(0, g, per) / torch.bincount(g, minlength=len(q)).clamp(min=1)
            present = torch.bincount(g, minlength=len(q)) > 0
            q = q * torch.exp(DRO_ETA * gl.detach() * present); q = q / q.sum()
            loss = (q * gl).sum()
        else:
            w = weights[idx].to(DEV)
            loss = (per * w).sum() / w.sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if step % 200 == 0:
            print(f"{variant} step {step}/{steps} loss {loss.item():.4f} {(step + 1) * BATCH / (time.time() - t0):.0f} ex/s", flush=True)
    model.eval()
    name = run_name(variant)
    out = MODELS / name
    model.save_pretrained(out); tok.save_pretrained(out)
    info = dict(variant=variant, base=BASE, seed=SEED, device=DEV, n_train=len(tr), steps=steps, minutes=(time.time() - t0) / 60)
    for split in ("val", "test"):
        s = d[d.split == split].head(300 if SMOKE else None)
        z = score(model, tok, s.text)
        pd.DataFrame(z, columns=LABELS).assign(id=s.id.values).to_parquet(PREDS / f"{name}_{split}.parquet")
    info["total_minutes"] = (time.time() - t0) / 60
    (out / "train_info.json").write_text(json.dumps(info))
    print(info)


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True); PREDS.mkdir(exist_ok=True)
    if sys.argv[1] == "selfcheck":  # IG completeness: attributions sum to f(x) - f(baseline)
        m = base_model().to(DEV).eval()
        t = AutoTokenizer.from_pretrained(TOKENIZER)
        w, s, delta = integrated_gradients(m, t, "Muslims are praying today, unbelievable", steps=200)
        print(w, s.round(3), s.sum(), delta)
        assert abs(s.sum() - delta) < 0.05 * max(1, abs(delta)), "IG completeness failed"
        print("IG self-check passed")
    else:
        for v in sys.argv[1:]:
            train(v)

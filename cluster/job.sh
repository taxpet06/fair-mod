#!/bin/bash
# Runs on a GPU node. stdin = tar of code + data/prepared.parquet; stdout = tar of models/ preds/; everything else -> stderr.
# GPU nodes can't read home or /jumbo, but have internet and a big /tmp (see ~/lab id8_stardist notes).
set -e
W=/tmp/fm_${SLURM_JOB_ID}; mkdir -p $W; cd $W
trap 'cd /; rm -rf $W' EXIT   # always clean the node's /tmp, even on failure
export HOME=$W HF_HOME=$W/hf XDG_CACHE_HOME=$W/cache   # home is unreadable from GPU nodes
tar xf - ; hostname >&2; nvidia-smi --query-gpu=name --format=csv,noheader >&2
/usr/bin/python3.12 -m venv venv >&2; . venv/bin/activate
pip -q install torch==2.15.0 transformers==5.22.2 pandas==3.1.1 pyarrow==26.0.1 scikit-learn==1.9.1 numpy==2.5.4 joblib >&2
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
python bert.py $VARIANT >&2
[ "$SEED" = "0" ] || rm -rf models/*   # replicate seeds: predictions only
tar cf - models preds

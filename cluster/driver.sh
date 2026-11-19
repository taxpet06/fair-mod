#!/bin/bash
# On the cluster login node: run every (variant, seed) as its own GPU job, at most 6 at a time, retrying preempted ones.
cd /scratch/$USER/fairmod; mkdir -p out logs
export BASE=distilbert/distilbert-base-uncased TOKENIZER=distilbert/distilbert-base-uncased TAG=distil LR=5e-5 N_TRAIN=0
run() {
  v=$1; s=$2; tag=${v}_s$s
  for try in 1 2 3; do
    tar cf - -C in . | VARIANT=$v SEED=$s srun -p gpu_preempt -A free --exclude=gv01,p[01-04],turing[01-02] --gres=gpu:1 --cpus-per-task=4 --mem=32G -t 02:00:00 -J fm_$tag \
      bash -c "$(cat job.sh)" 2> logs/$tag.try$try.log | tar xf - -C out 2>/dev/null
    [ -f out/preds/bert_distil_${v}$( [ $s = 0 ] || echo _s$s )_test.parquet ] && { echo "$tag ok (try $try)"; return; }
    echo "$tag failed try $try"
  done
}
export -f run
# JOBS="v s v s ..." reruns a subset; default is the full grid
${JOBS:+false} && JOBS=$(for s in 0 1 2; do for v in erm cda reweight dro soft combo; do echo "$v $s"; done; done)
echo $JOBS | xargs -P 6 -n 2 bash -c 'run "$@"' _
echo ALLDONE

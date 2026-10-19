#!/usr/bin/env bash
# Run from any directory. Never reuse an output directory, even for resume.
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "Usage: bash scripts/maple/run_gating.sh DATA_ROOT {oxford_pets|dtd} {none|scalar|channel} SEED RUN_ID {smoke|formal}"
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
data=$1 dataset=$2 mode=$3 seed=$4 run_id=$5 stage=$6
case "$dataset" in oxford_pets|dtd) ;; *) echo 'Unsupported dataset'; exit 2;; esac
case "$mode" in none|scalar|channel) ;; *) echo 'Invalid gate mode'; exit 2;; esac
[[ "$seed" =~ ^[1-9][0-9]*$ && "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Use a positive seed and a valid run ID'; exit 2; }
[[ -d "$data/$dataset" ]] || { echo "Missing dataset: $data/$dataset"; exit 2; }
extra=()
case "$stage" in
  smoke) epoch=1; extra=(OPTIM.MAX_EPOCH 1 OPTIM.WARMUP_EPOCH 0 DATALOADER.NUM_WORKERS 0 DATALOADER.TEST.BATCH_SIZE 4);;
  formal) epoch=5;;
  *) echo 'Invalid stage'; exit 2;;
esac
python -c 'import torch; assert torch.cuda.is_available(), "Requires GPU mode; no training launched"'
run="output/gating_v1/$stage/$run_id/$dataset/$mode/seed$seed"
mkdir -p "$(dirname "$run")"
mkdir "$run"  # atomic refusal to overwrite or automatically resume
git rev-parse HEAD > "$run/base_commit.txt"
git diff --binary > "$run/source.patch"
tar --exclude='__pycache__' -czf "$run/source.tar.gz" train.py trainers clip configs scripts/maple tests
common=(--root "$data" --seed "$seed" --trainer MaPLe
  --dataset-config-file "configs/datasets/$dataset.yaml"
  --config-file "configs/trainers/MaPLe/vit_b16_gating_$mode.yaml")
python scripts/maple/profile_run.py "$run/train_metrics.json" "${common[@]}" \
  --output-dir "$run/train_base" DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base "${extra[@]}"
for subset in base new; do
  python scripts/maple/profile_run.py "$run/test_${subset}_metrics.json" "${common[@]}" \
    --output-dir "$run/test_$subset" --eval-only --model-dir "$run/train_base" --load-epoch "$epoch" \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES "$subset" "${extra[@]}"
done
python scripts/maple/summarize_gating.py "$run"

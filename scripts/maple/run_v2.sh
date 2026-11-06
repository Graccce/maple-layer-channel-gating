#!/usr/bin/env bash
# Phase-2 runner: one MaPLe variant = base gating config + explicit TRAINER/OPTIM overrides.
# Usage: bash scripts/maple/run_v2.sh DATA_ROOT DATASET TAG SEED RUN_ID {smoke|formal} [KEY VALUE ...]
#   TAG names the variant directory; overrides are passed verbatim to train.py for train AND eval.
#   KEEP_CKPT=1 keeps the (268 MB) epoch checkpoint; by default it is hashed then deleted.
set -euo pipefail
if [[ $# -lt 6 ]]; then
  echo "Usage: bash scripts/maple/run_v2.sh DATA_ROOT DATASET TAG SEED RUN_ID {smoke|formal} [KEY VALUE ...]"
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
data=$1 dataset=$2 tag=$3 seed=$4 run_id=$5 stage=$6
shift 6
overrides=("$@")
(( ${#overrides[@]} % 2 == 0 )) || { echo 'Overrides must be KEY VALUE pairs'; exit 2; }
case "$dataset" in
  oxford_pets) folder=$dataset split=split_zhou_OxfordPets.json;;
  dtd) folder=$dataset split=split_zhou_DescribableTextures.json;;
  oxford_flowers) folder=$dataset split=split_zhou_OxfordFlowers.json;;
  eurosat) folder=$dataset split=split_zhou_EuroSAT.json;;
  fgvc_aircraft) folder=$dataset split=images_variant_train.txt;;
  caltech101) folder=caltech-101 split=split_zhou_Caltech101.json;;
  *) echo 'Unsupported dataset'; exit 2;;
esac
[[ "$seed" =~ ^[1-9][0-9]*$ && "$run_id" =~ ^[a-zA-Z0-9_-]+$ && "$tag" =~ ^[a-zA-Z0-9_.-]+$ ]] \
  || { echo 'Use a positive seed and valid run ID / tag'; exit 2; }
[[ -f "$data/$folder/$split" ]] || { echo "Missing fixed split: $data/$folder/$split"; exit 2; }
extra=()
case "$stage" in
  smoke) epoch=1; extra=(OPTIM.MAX_EPOCH 1 OPTIM.WARMUP_EPOCH 0);;
  formal) epoch=5;;
  *) echo 'Invalid stage'; exit 2;;
esac
python -c 'import torch; assert torch.cuda.is_available(), "Requires GPU mode; no training launched"'
run="output/v2/$stage/$run_id/$dataset/$tag/seed$seed"
mkdir -p "$(dirname "$run")"
mkdir "$run"  # atomic refusal to overwrite or automatically resume
git rev-parse HEAD > "$run/base_commit.txt" 2>/dev/null || cat SOURCE_COMMIT > "$run/base_commit.txt"
printf '%s\n' "${overrides[@]}" > "$run/overrides.txt"
tar --exclude='__pycache__' -czf "$run/source.tar.gz" train.py trainers clip configs scripts/maple tests
common=(--root "$data" --seed "$seed" --trainer MaPLe
  --dataset-config-file "configs/datasets/$dataset.yaml"
  --config-file "configs/trainers/MaPLe/vit_b16_gating_none.yaml")
python scripts/maple/profile_run.py "$run/train_metrics.json" "${common[@]}" \
  --output-dir "$run/train_base" DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base "${extra[@]}" "${overrides[@]}"
for subset in base new; do
  python scripts/maple/profile_run.py "$run/test_${subset}_metrics.json" "${common[@]}" \
    --output-dir "$run/test_$subset" --eval-only --model-dir "$run/train_base" --load-epoch "$epoch" \
    DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES "$subset" "${extra[@]}" "${overrides[@]}"
done
python scripts/maple/summarize_gating.py "$run" > /dev/null
ckpt="$run/train_base/MultiModalPromptLearner/model.pth.tar-$epoch"
sha256sum "$ckpt" > "$run/checkpoint.sha256"
[[ "${KEEP_CKPT:-0}" == 1 ]] || rm -f "$ckpt"
cat "$run/results.json"

#!/usr/bin/env bash
# Sequential phase-2 job queue (the exact queues used are in scripts/maple/queues_v2/).
# Each non-comment line: "DATASET TAG SEED RUN_ID STAGE [KEY VALUE ...]", passed to run_v2.sh.
# A failed run is logged and the queue continues; results never overwrite (run_v2.sh refuses).
# Seed-1 checkpoints of cond_channel / none_sanity are kept (gate statistics, bitwise baseline check).
# Usage: DATA_ROOT=/path/to/data bash scripts/maple/run_queue.sh scripts/maple/queues_v2/01_main.txt
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root}"
queue=$1
mkdir -p logs
log=logs/$(basename "$queue" .txt).log
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  set -- $line
  echo "[$(date +%H:%M:%S)] START $line" >> "$log"
  keep=0
  [[ "$3" == 1 && ( "$2" == cond_channel || "$2" == none_sanity ) ]] && keep=1
  if KEEP_CKPT=$keep bash scripts/maple/run_v2.sh "$DATA_ROOT" "$@" > logs/last_run.out 2>&1; then
    hm=$(tr -d '\n ' < "output/v2/$5/$4/$1/$2/seed$3/results.json" | grep -o '"harmonic_mean":[0-9.]*')
    echo "[$(date +%H:%M:%S)] DONE  $line $hm" >> "$log"
  else
    cp logs/last_run.out "logs/fail_$1_$2_$3_$4.out"
    echo "[$(date +%H:%M:%S)] FAIL  $line (see logs/fail_$1_$2_$3_$4.out)" >> "$log"
  fi
done < "$queue"
echo "[$(date +%H:%M:%S)] QUEUE FINISHED" >> "$log"

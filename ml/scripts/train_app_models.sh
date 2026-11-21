#!/usr/bin/env bash
# Train the models behind the app on MovieLens-32M (2018-2023) and export the serving bundle.
# Runs one job at a time; models already trained with identical settings are skipped (settings hash).
#   bash ml/scripts/train_app_models.sh            first run (log: data/logs/train_app_models.log)
#   bash ml/scripts/train_app_models.sh --refresh  after `recsys import-app-ratings`: rebuilds the data
#                                                  with the app's ratings and retrains every model
# On a 4-core laptop CPU: about 20 minutes for EASE, iALS, Markov and MostPop, then 4-5 hours for SASRec.
# With --refresh the split changes, so its fingerprint changes and nothing is skipped: plan for the full time.
set -u
REFRESH=0
case "${1:-}" in
  "") ;;
  --refresh) REFRESH=1 ;;
  *) echo "usage: bash ml/scripts/train_app_models.sh [--refresh]" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.." || exit 1
mkdir -p ../data/logs
LOG=../data/logs/train_app_models.log
DATA=../data/processed/ml-32m
export PYTHONIOENCODING=utf-8
run() {
  echo "=== $(date '+%F %T') START recsys $*" >> "$LOG"
  uv run --no-sync recsys "$@" >> "$LOG" 2>&1
  local code=$?
  echo "=== $(date '+%F %T') END   recsys $* (exit $code)" >> "$LOG"
  return $code
}
# missing, or --refresh: the film list can change when app ratings join, so links and texts follow it
needed() { [ "$REFRESH" = 1 ] || [ ! -f "$DATA/$1" ]; }
needed interactions.parquet && { run prepare ml-32m || exit 1; }
needed links.parquet && { run tmdb ml-32m --links-only || exit 1; }
needed content.parquet && { run content ml-32m || exit 1; }
run split -c ml-32m || exit 1  # remade automatically when the data is newer than the split
run benchmark -c ml-32m || exit 1
run export -c ml-32m || exit 1
echo "=== $(date '+%F %T') DONE" >> "$LOG"

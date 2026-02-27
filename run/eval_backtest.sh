#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

Rscript src/eval_backtest.R \
  --hub-dir "${ROOT_DIR}/RespiCast-SyndromicIndicators" \
  --canonical-data "${ROOT_DIR}/data/processed/respicast_long_latest.csv" \
  --forecast-dir "${ROOT_DIR}/forecasts/retrospective/submission" \
  --output-dir "${ROOT_DIR}/forecasts/retrospective/evaluation/tables" \
  --output-file "${ROOT_DIR}/forecasts/retrospective/evaluation/latest-forecast_scores.csv" \
  --start-origin-date "2025-10-01" \
  --baseline-model "respicast-quantileBaseline" \
  --recent-weeks "4" \
  --write-row-level "false" \
  "$@"

Rscript -e "rmarkdown::render('src/eval_backtest.Rmd', output_file='eval_backtest.html', output_dir='forecasts/retrospective/evaluation', knit_root_dir=getwd(), params=list(tables_dir='forecasts/retrospective/evaluation/tables'))"

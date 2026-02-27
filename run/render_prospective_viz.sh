#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

Rscript -e "rmarkdown::render('src/prospective_joint_twostage_viz.Rmd', output_file='prospective_joint_twostage_viz.html', output_dir='forecasts/prospective/viz')"

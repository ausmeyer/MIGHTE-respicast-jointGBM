# MIGHTE RespiCast Joint Two-Stage Pipeline

Prospective-only forecasting pipeline for European RespiCast ILI/ARI targets, adapted from the US joint two-stage approach with no imputation.

## Directory layout

- `run/`
  - `run_weekly.sh`: builds canonical data and generates weekly prospective forecasts for ILI + ARI.
  - `run_prospective.sh`: direct entrypoint to `src/forecast_prospective.py`.
  - `render_prospective_viz.sh`: renders the interactive dashboard HTML.
- `src/`
  - `build_long_timeseries.py`: merges `latest-*` and snapshot files into one canonical long series.
  - `model_joint_twostage_eu.py`: joint two-stage pooled-horizon model (prospective mode).
  - `forecast_prospective.py`: orchestration for canonical data build + ILI/ARI forecast generation.
  - `prospective_joint_twostage_viz.Rmd`: interactive visualization over all saved submissions.
- `data/processed/`
  - `respicast_long_latest.csv`: canonical merged long-series input cache.
- `forecasts/prospective/submission/`
  - Weekly forecast files using model-output style naming: `YYYY-MM-DD-<model-id>.csv`.
  - Re-running for the same reference date overwrites that same file.
- `forecasts/prospective/raw/`
  - Optional duplicate raw output (only when `--save-raw` is set).

## Data precedence

Canonical long-series merge rule by `(target, location, truth_date)`:
1. `latest-*.csv` rows take precedence.
2. Otherwise, newer snapshots take precedence.
3. If still tied, source priority is `ERVISS` then `FluID`.

## Quick start

From `MIGHTE-respicast-jointGBM/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run/run_weekly.sh
./run/render_prospective_viz.sh
```

`run_weekly.sh` forwards extra CLI flags to `src/forecast_prospective.py`, e.g.:

```bash
./run/run_weekly.sh --num-bags 20 --stage1-rounds 100 --stage2-rounds 80
```

Default location scope is the 17-location ILI wide-file set:
`AT,BE,CZ,EE,FI,FR,GR,HU,IE,IS,LT,LU,MT,NL,NO,PL,RO`

Dashboard output:
- `forecasts/prospective/viz/prospective_joint_twostage_viz.html`

## Output format

Submission files use RespiCast model-output schema:

- `origin_date`
- `target`
- `target_end_date`
- `horizon` (1..4)
- `location`
- `output_type` (`quantile`)
- `output_type_id`
- `value`

Filename/reference-date convention:
- The reference date is the date in the filename prefix (`YYYY-MM-DD`).
- The dashboard uses this filename date as `origin_date`.

## Failure behavior

The prospective model run is strict:
- If no LightGBM/LightGBMLSS bags fit successfully, the pipeline errors.
- If any location/horizon pair is missing model predictions, the pipeline errors.
- Locations with no observed truth in the last 4 weeks are excluded from forecasting
  (configurable with `--recent-weeks-required`).

There is no persistence fallback.

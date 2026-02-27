#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(purrr)
})

parse_args <- function(argv) {
  out <- list(
    hub_dir = "RespiCast-SyndromicIndicators",
    canonical_data = "data/processed/respicast_long_latest.csv",
    forecast_dir = "forecasts/retrospective/submission",
    output_dir = "forecasts/retrospective/evaluation/tables",
    output_file = "forecasts/retrospective/evaluation/latest-forecast_scores.csv",
    start_origin_date = "2025-10-01",
    baseline_model = "respicast-quantileBaseline",
    recent_weeks = "4",
    write_row_level = "false"
  )
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) {
      stop("Unexpected argument: ", key)
    }
    if (i == length(argv)) {
      stop("Missing value for argument: ", key)
    }
    val <- argv[[i + 1L]]
    nm <- gsub("^--", "", key)
    nm <- gsub("-", "_", nm)
    if (!nm %in% names(out)) {
      stop("Unknown argument: ", key)
    }
    out[[nm]] <- val
    i <- i + 2L
  }
  out
}

as_bool <- function(x) {
  tolower(as.character(x)) %in% c("1", "true", "t", "yes", "y")
}

extract_origin <- function(path) {
  nm <- basename(path)
  m <- str_match(nm, "^(\\d{4}-\\d{2}-\\d{2})-")
  as.Date(m[, 2])
}

extract_model <- function(path) {
  nm <- basename(path)
  nm <- str_remove(nm, "\\.csv$")
  str_remove(nm, "^\\d{4}-\\d{2}-\\d{2}-")
}

split_team_model <- function(model) {
  m <- str_match(model, "^([^-]+)-(.+)$")
  team <- ifelse(is.na(m[, 2]), model, m[, 2])
  mdl <- ifelse(is.na(m[, 3]), model, m[, 3])
  tibble(team_id = team, model_id = mdl)
}

compute_wis <- function(probs, vals, truth, median_value) {
  if (is.na(truth) || is.na(median_value)) {
    return(NA_real_)
  }
  if (length(probs) != length(vals) || length(probs) == 0) {
    return(NA_real_)
  }

  probs <- as.numeric(probs)
  vals <- as.numeric(vals)
  o <- order(probs)
  probs <- probs[o]
  vals <- vals[o]

  lower_probs <- probs[probs < 0.5]
  if (length(lower_probs) == 0) {
    return(NA_real_)
  }

  interval_terms <- c()
  for (lp in lower_probs) {
    up <- 1 - lp
    li <- which(abs(probs - lp) < 1e-10)
    ui <- which(abs(probs - up) < 1e-10)
    if (length(li) == 0 || length(ui) == 0) {
      next
    }
    lq <- vals[li[1]]
    uq <- vals[ui[1]]
    alpha <- 2 * lp
    if (alpha <= 0) {
      next
    }
    is_alpha <- (uq - lq) +
      (2 / alpha) * (lq - truth) * as.numeric(truth < lq) +
      (2 / alpha) * (truth - uq) * as.numeric(truth > uq)
    interval_terms <- c(interval_terms, (alpha / 2) * is_alpha)
  }

  k <- length(interval_terms)
  if (k == 0) {
    return(NA_real_)
  }
  (0.5 * abs(truth - median_value) + sum(interval_terms)) / (k + 0.5)
}

load_forecasts <- function(files, model_override = NULL) {
  parts <- lapply(files, function(path) {
    origin <- extract_origin(path)
    if (is.na(origin)) {
      return(NULL)
    }
    model <- if (is.null(model_override)) {
      extract_model(path)
    } else if (is.function(model_override)) {
      model_override(path)
    } else {
      model_override
    }
    df <- read_csv(path, show_col_types = FALSE)
    req <- c("target", "target_end_date", "horizon", "location", "output_type", "value")
    if (!all(req %in% names(df))) {
      return(NULL)
    }
    if (!("output_type_id" %in% names(df))) {
      df$output_type_id <- NA_real_
    }
    df %>%
      mutate(
        origin_date = origin,
        target_end_date = as.Date(target_end_date),
        horizon = as.integer(horizon),
        location = as.character(location),
        target = as.character(target),
        output_type = tolower(as.character(output_type)),
        output_type_id = suppressWarnings(as.numeric(output_type_id)),
        value = as.numeric(value),
        model = model
      ) %>%
      filter(
        output_type %in% c("quantile", "median"),
        horizon %in% 1:4,
        !is.na(target_end_date),
        !is.na(value)
      )
  })
  bind_rows(parts)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

hub_dir <- normalizePath(args$hub_dir, mustWork = TRUE)
canonical_data <- normalizePath(args$canonical_data, mustWork = TRUE)
forecast_dir <- normalizePath(args$forecast_dir, mustWork = TRUE)
output_file <- args$output_file
output_dir <- args$output_dir
start_origin <- as.Date(args$start_origin_date)
baseline_model <- args$baseline_model
recent_weeks <- as.integer(args$recent_weeks)
write_row_level <- as_bool(args$write_row_level)

forecast_files <- list.files(
  forecast_dir,
  pattern = "^[0-9]{4}-[0-9]{2}-[0-9]{2}-.+\\.csv$",
  full.names = TRUE
)
forecast_files <- forecast_files[extract_origin(forecast_files) >= start_origin]
fc_local <- load_forecasts(forecast_files)

hub_model_output_dir <- file.path(hub_dir, "model-output")
hub_files <- list.files(
  hub_model_output_dir,
  recursive = TRUE,
  pattern = "^[0-9]{4}-[0-9]{2}-[0-9]{2}-.+\\.csv$",
  full.names = TRUE
)
hub_files <- hub_files[extract_origin(hub_files) >= start_origin]
fc_hub <- load_forecasts(
  hub_files,
  model_override = function(path) basename(dirname(path))
)

fc_main <- bind_rows(fc_local, fc_hub)
if (nrow(fc_main) == 0) {
  stop(
    "No valid forecasts found in retrospective folder or hub model-output ",
    "for start date ", start_origin
  )
}

fc_all <- fc_main %>%
  distinct(
    model, origin_date, target, target_end_date, horizon, location, output_type, output_type_id,
    .keep_all = TRUE
  )

truth <- read_csv(canonical_data, show_col_types = FALSE) %>%
  transmute(
    target = as.character(target),
    location = as.character(location),
    target_end_date = as.Date(truth_date),
    truth_value = as.numeric(value)
  ) %>%
  group_by(target, location, target_end_date) %>%
  summarise(truth_value = mean(truth_value, na.rm = TRUE), .groups = "drop")

keys <- c("model", "origin_date", "target", "target_end_date", "horizon", "location")

quant <- fc_all %>%
  filter(output_type == "quantile", !is.na(output_type_id)) %>%
  group_by(across(all_of(keys))) %>%
  summarise(
    probs = list(output_type_id),
    vals = list(value),
    q50 = value[which.min(abs(output_type_id - 0.5))],
    .groups = "drop"
  )

median_only <- fc_all %>%
  filter(output_type == "median") %>%
  group_by(across(all_of(keys))) %>%
  summarise(median_only = first(value), .groups = "drop")

scored_base <- quant %>%
  full_join(median_only, by = keys) %>%
  mutate(median_value = coalesce(q50, median_only)) %>%
  left_join(truth, by = c("target", "location", "target_end_date")) %>%
  filter(!is.na(truth_value))

scored_base <- scored_base %>%
  mutate(
    wis = pmap_dbl(
      list(probs, vals, truth_value, median_value),
      function(probs, vals, truth_value, median_value) {
        compute_wis(probs, vals, truth_value, median_value)
      }
    ),
    ae = abs(median_value - truth_value)
  )

scores <- bind_rows(
  scored_base %>%
    transmute(
      origin_date, target, target_end_date, horizon, location, model,
      metric = "WIS", value_absolute = wis
    ),
  scored_base %>%
    transmute(
      origin_date, target, target_end_date, horizon, location, model,
      metric = "AE", value_absolute = ae
    )
) %>%
  filter(!is.na(value_absolute))

team_model <- split_team_model(scores$model)
scores <- bind_cols(scores, team_model) %>%
  select(
    origin_date, target, target_end_date, horizon, location, team_id, model_id, metric, value_absolute
  )

baseline_split <- split_team_model(baseline_model)
baseline_team <- baseline_split$team_id[[1]]
baseline_model_id <- baseline_split$model_id[[1]]

if (!any(scores$team_id == baseline_team & scores$model_id == baseline_model_id)) {
  stop(
    "Baseline model '", baseline_model,
    "' not found among scored forecasts in the selected period"
  )
}

group_keys <- c("origin_date", "target", "target_end_date", "horizon", "location", "metric")
scores <- scores %>%
  group_by(across(all_of(group_keys))) %>%
  mutate(
    baseline_score = {
      idx <- which(team_id == baseline_team & model_id == baseline_model_id)
      if (length(idx) == 0) NA_real_ else value_absolute[idx[1]]
    },
    value_relative = ifelse(
      !is.na(baseline_score) & baseline_score > 0 & value_absolute > 0,
      log2(baseline_score / value_absolute),
      NA_real_
    ),
    n_models = sum(!is.na(value_absolute)),
    rank = rank(value_absolute, ties.method = "min", na.last = "keep"),
    rank_score = {
      v <- value_absolute
      rng <- range(v, na.rm = TRUE)
      if (!all(is.finite(rng))) {
        rep(NA_real_, length(v))
      } else if (abs(rng[2] - rng[1]) < 1e-12) {
        rep(1, length(v))
      } else {
        (rng[2] - v) / (rng[2] - rng[1])
      }
    }
  ) %>%
  ungroup() %>%
  select(
    origin_date, target, target_end_date, horizon, location, team_id, model_id, metric,
    value_absolute, value_relative, n_models, rank, rank_score
  ) %>%
  arrange(origin_date, target, target_end_date, horizon, location, metric, team_id, model_id)

unit_scores <- scores %>%
  mutate(model = paste(team_id, model_id, sep = "-")) %>%
  select(
    origin_date, target, target_end_date, horizon, location,
    team_id, model_id, model, metric, value_absolute, value_relative
  ) %>%
  pivot_wider(
    names_from = metric,
    values_from = c(value_absolute, value_relative),
    names_sep = "__"
  ) %>%
  transmute(
    origin_date = as.Date(origin_date),
    target,
    target_end_date,
    horizon,
    location,
    team_id,
    model_id,
    model,
    wis = value_absolute__WIS,
    ae = value_absolute__AE,
    rel_wis_log2 = value_relative__WIS
  )

if (nrow(unit_scores) == 0) {
  stop("No scored backtest units found after filtering to observed truth")
}

summarise_table <- function(df, group_cols, rank_group_cols) {
  out <- df %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      mean_wis = mean(wis, na.rm = TRUE),
      mean_ae = mean(ae, na.rm = TRUE),
      mean_rel_wis_log2 = mean(rel_wis_log2, na.rm = TRUE),
      n_units = sum(!is.na(wis)),
      .groups = "drop"
    )
  out %>%
    group_by(across(all_of(rank_group_cols))) %>%
    mutate(
      rank = ifelse(
        is.na(mean_rel_wis_log2),
        NA_integer_,
        dense_rank(desc(mean_rel_wis_log2))
      )
    ) %>%
    ungroup() %>%
    arrange(across(all_of(rank_group_cols)), rank, model)
}

last_origin <- max(unit_scores$origin_date, na.rm = TRUE)
recent_start <- last_origin - (7 * recent_weeks)

season_units <- unit_scores %>%
  filter(origin_date >= start_origin, origin_date <= last_origin)
recent_units <- season_units %>%
  filter(origin_date >= recent_start)

season_per_horizon <- summarise_table(
  season_units,
  group_cols = c("target", "horizon", "team_id", "model_id", "model"),
  rank_group_cols = c("target", "horizon")
)
season_overall <- summarise_table(
  season_units,
  group_cols = c("target", "team_id", "model_id", "model"),
  rank_group_cols = c("target")
)
recent_per_horizon <- summarise_table(
  recent_units,
  group_cols = c("target", "horizon", "team_id", "model_id", "model"),
  rank_group_cols = c("target", "horizon")
)
recent_overall <- summarise_table(
  recent_units,
  group_cols = c("target", "team_id", "model_id", "model"),
  rank_group_cols = c("target")
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(
  recent_per_horizon,
  file.path(output_dir, "recent_per_horizon.csv")
)
write_csv(
  recent_overall,
  file.path(output_dir, "recent_overall.csv")
)
write_csv(
  season_per_horizon,
  file.path(output_dir, "season_per_horizon.csv")
)
write_csv(
  season_overall,
  file.path(output_dir, "season_overall.csv")
)

meta <- tibble(
  start_origin_date = start_origin,
  last_origin_date = last_origin,
  recent_start_date = recent_start,
  recent_weeks = recent_weeks,
  baseline_model = baseline_model,
  n_scored_units = nrow(unit_scores)
)
write_csv(meta, file.path(output_dir, "evaluation_metadata.csv"))

cat("Wrote table outputs to:", output_dir, "\n")
cat("  - recent_per_horizon.csv\n")
cat("  - recent_overall.csv\n")
cat("  - season_per_horizon.csv\n")
cat("  - season_overall.csv\n")
cat("  - evaluation_metadata.csv\n")

if (write_row_level) {
  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  write_csv(scores, output_file)
  cat("Wrote row-level scores:", output_file, "rows=", nrow(scores), "\n")
}

#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(readr)
library(yardstick)

read_csv_safe <- function(path) {
  read_csv(path, col_types = list(
    afa_method = col_factor(),
    classifier = col_factor(),
    dataset = col_factor(),
    selections_performed = col_integer(),
    features_observed = col_integer(),
    predicted_class = col_factor(),
    true_class = col_factor(),
    train_seed = col_integer(),
    eval_seed = col_integer(),
    train_hard_budget = col_integer(),
    eval_hard_budget = col_integer(),
    train_soft_budget_param = col_number(),
    eval_soft_budget_param = col_number()
  ))
}

generate_dummy_data <- function(n = 100) {
  set.seed(42)
  methods <- c("ODIN-MFRL", "EDDI", "DIME")
  classifiers <- c(NA, "fully_connected")
  datasets <- c("cube", "MNIST")
  train_seeds <- 1:2
  eval_seeds <- 1:2
  train_hard_budgets <- c(NA_integer_, 5, 10, 15)
  eval_hard_budgets <- c(NA_integer_, 5, 10, 15)
  train_soft_budget_params <- c(NA, 0.1, 1.0)
  eval_soft_budget_params <- c(NA, 0.1, 1.0)
  n_classes <- 5
  n_features <- 15

  df <- tibble(
    afa_method = factor(sample(methods, n, replace = TRUE)),
    classifier = factor(sample(classifiers, n, replace = TRUE)),
    dataset = factor(sample(datasets, n, replace = TRUE)),
    selections_performed = sample(1:n_features, n, replace = TRUE),
    features_observed = selections_performed,
    predicted_class = factor(sample(0:(n_classes - 1), n, replace = TRUE)),
    true_class = factor(sample(0:(n_classes - 1), n, replace = TRUE)),
    train_seed = sample(train_seeds, n, replace = TRUE),
    eval_seed = sample(eval_seeds, n, replace = TRUE),
    train_hard_budget = as.integer(sample(train_hard_budgets, n, replace = TRUE)),
    eval_hard_budget = as.integer(sample(eval_hard_budgets, n, replace = TRUE)),
    train_soft_budget_param = sample(train_soft_budget_params, n, replace = TRUE),
    eval_soft_budget_param = sample(eval_soft_budget_params, n, replace = TRUE)
  )

  return(df)
}


# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 1) {
  print("Proceeding with plotting using dummy data...")
  df <- generate_dummy_data(n=10000)
} else if (length(args) == 2) {
  df <- read_csv_safe(args[1])
} else {
  stop("Usage: Rscript plot.R eval_results.csv output_folder")
}

class_metrics <- metric_set(
  accuracy,
  kap,
  f_meas
)

# First type of plot: hard budget
df_hard_budget <- df |>
  # We filter results to only include methods that were not trained/evaluated with a soft budget parameter.
  filter(
    !is.na(eval_hard_budget),
    is.na(train_soft_budget_param),
    is.na(eval_soft_budget_param),
  ) |>
  # We don't distinguish between training hard budget and evaluation hard budget
  select(
    -train_hard_budget
  ) |>
  group_by(
    afa_method,
    classifier,
    dataset,
  ) |>
  select(
    -train_soft_budget_param,
    -eval_soft_budget_param
  ) |>
  # We only care about the prediction made once we reach the evaluation hard budget
  filter(
    features_observed == eval_hard_budget
  ) |>
  select(
    -features_observed,
    -selections_performed
  ) |>
  # Also group over train_seed, eval_seed and eval_hard_budget before calculating metrics.
  # eval_hard_budget will become the x-axis
  group_by(
    train_seed,
    eval_seed,
    eval_hard_budget,
    .add = TRUE
  ) |>
  # Now we can calculate metrics
  class_metrics(truth=true_class, estimate=predicted_class) |>
  # tibble is automatically ungrouped after calculating metrics
  # Calculate mean and std of metrics
  group_by(
    afa_method,
    classifier,
    dataset,
    eval_hard_budget,
    .metric,
    .estimator
    # leaves train_seed, eval_seed and .estimate
  ) |>
  summarize(
    estimate_mean = mean(.estimate),
    estimate_sd = sd(.estimate)
  )



# Example: plot with builtin classifier accuracy
df_hard_budget |>
  filter(
    .metric == "accuracy",
    is.na(classifier)
  ) |>
  ggplot(aes(x = eval_hard_budget, y = estimate_mean, color = afa_method, fill = afa_method)) +
  geom_line() +
  geom_ribbon(aes(ymin = estimate_mean - estimate_sd, ymax = estimate_mean + estimate_sd), alpha=0.2) +
  facet_wrap(vars(dataset))

# Second type of plot: soft budget
df_soft_budget <- df |>
  # We filter results to only include methods that were trained or evaluated with a soft budget parameter.
  # Most methods either use a parameter at training time or evaluation time, but there could in theory
  # be methods that accept both types
  filter(
    is.na(eval_hard_budget),
    !is.na(train_soft_budget_param) | !is.na(eval_soft_budget_param)
  ) |>
  # We don't distinguish between training hard budget and evaluation hard budget
  select(
    -train_hard_budget
  ) |>
  group_by(
    afa_method,
    classifier,
    dataset,
  ) |>
  select(
    -eval_hard_budget
  ) |>
  # This time we care about predictions at every step, but we focus on the number of features observed instead of selections performed
  select(
    -selections_performed
  ) |>
  # Also group over (train_seed, eval_seed, train_soft_budget_param, eval_soft_budget_param, features_observed) before calculating metrics.
  # eval_hard_budget will become the x-axis
  group_by(
    train_seed,
    eval_seed,
    train_soft_budget_param,
    eval_soft_budget_param,
    features_observed,
    .add = TRUE
  ) |>
  # Now we can calculate metrics
  class_metrics(truth=true_class, estimate=predicted_class) |>
  # tibble is automatically ungrouped after calculating metrics
  # Calculate mean and std of metrics
  group_by(
    afa_method,
    classifier,
    dataset,
    features_observed,
    .metric,
    .estimator
    # leaves train_seed, eval_seed, train_soft_budget_param, eval_soft_budget_param and .estimate
  ) |>
  summarize(
    estimate_mean = mean(.estimate),
    estimate_sd = sd(.estimate),
    .groups = "drop"
  )

# Example: plot with builtin classifier accuracy
df_soft_budget |>
  filter(
    .metric == "accuracy",
    is.na(classifier)
  ) |>
  ggplot(aes(x = features_observed, y = estimate_mean, color = afa_method, fill = afa_method)) +
  geom_line() +
  geom_ribbon(aes(ymin = estimate_mean - estimate_sd, ymax = estimate_mean + estimate_sd), alpha=0.2) +
  facet_wrap(vars(dataset))

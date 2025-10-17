#!/usr/bin/env Rscript

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: Rscript produce_soft_budget_plots.R eval_results.csv output_plot1.png output_plot2.png cost_params_plot.png")
}

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(knitr)
library(kableExtra)

read_single_csv <- function(path) {
  read_csv(path, col_types = list(
    features_chosen = col_integer(),
    predicted_label_external = col_integer(),
    true_label = col_integer(),
    predicted_label_builtin = col_integer(),
    acquisition_cost = col_double(),
    method = col_factor(),
    training_seed = col_integer(),
    cost_parameter = col_double(),
    dataset = col_factor(),
    dataset_split = col_integer()
  ))
}

read_multiple_csv <- function(paths) {
  paths |>
    lapply(read_single_csv) |>
    bind_rows()
}

verify_df <- function(df) {
  # Specify your expected columns and types
  expected_types <- cols(
    method = col_factor(),
    training_seed = col_integer(),
    cost_parameter = col_double(),
    dataset = col_factor(),
    dataset_split = col_integer(),
    features_chosen = col_integer(),
    acquisition_cost = col_double(),
    predicted_label_builtin = col_integer(),
    predicted_label_external = col_integer(),
    true_label = col_integer()
  )

  # Check for unspecified columns
  unspecified_cols <- setdiff(names(df), names(expected_types$cols))
  if (length(unspecified_cols) > 0) {
    stop(
      paste(
        "The following columns were not specified in col_types:",
        paste(unspecified_cols, collapse = ", ")
      )
    )
  }
}

tidy_df <- function(df) {
  # Rename methods
  df <- df |>
    mutate(method = recode(method,
      "aaco" = "AACO",
      "covert2023" = "GDFS",
      "gadgil2023" = "DIME",
      "kachuee2019" = "OL-MFRL",
      "ma2018" = "EDDI",
      "shim2018" = "JAFA-MFRL",
      "zannone2019" = "ODIN-MFRL",
      "randomdummy" = "Random"
    ))
  # Tidy data
  df <- df %>%
    pivot_longer(
      cols = c(predicted_label_external, predicted_label_builtin),
      names_to = "prediction_type",
      names_prefix = "predicted_label_",
      values_to = "predicted_label"
    ) %>%
    filter(!is.na(predicted_label)) # Remove rows where builtin is NA
  df
}

summarize_df <- function(df) {
  # Summarize accuracy and features chosen
  df_summarized <- df %>%
    mutate(
      correct = (predicted_label == true_label),
      tp = (predicted_label == 1 & true_label == 1),
      fp = (predicted_label == 1 & true_label == 0),
      tn = (predicted_label == 0 & true_label == 0),
      fn = (predicted_label == 0 & true_label == 1)
    ) %>%
    group_by(method, prediction_type, dataset, dataset_split, cost_parameter, training_seed) %>%
    summarize(
      accuracy = mean(correct),
      avg_features_chosen = mean(features_chosen),
      avg_acquisition_cost = mean(acquisition_cost),
      tp = sum(tp),
      fp = sum(fp),
      tn = sum(tn),
      fn = sum(fn),
      precision = ifelse(tp + fp == 0, 0, tp / (tp + fp)),
      recall = ifelse(tp + fn == 0, 0, tp / (tp + fn)),
      f1 = ifelse(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall)),
      .groups = "keep"
    )

  # Some datasets use accuracy, others use F1
  df_summarized <- df_summarized %>%
    mutate(
      metric_type = ifelse(dataset %in% f1_datasets, "f1", "accuracy"),
      metric_value = ifelse(metric_type == "f1", f1, accuracy)
    )

  # Mean and sd over dataset splits and training seeds
  df_summary <- df_summarized %>%
    group_by(method, prediction_type, dataset, cost_parameter, metric_type) %>%
    summarize(
      avg_metric = mean(metric_value),
      sd_metric = sd(metric_value),
      mean_avg_features_chosen = mean(avg_features_chosen),
      sd_avg_features_chosen = sd(avg_features_chosen),
      mean_avg_acquisition_cost = mean(avg_acquisition_cost),
      sd_avg_acquisition_cost = sd(avg_acquisition_cost),
      .groups = "drop"
    )

  df_summary
}

unclutter_df <- function(df) {
  df |>
    group_by(dataset) |>
    # Only two datasets
    filter(dataset %in% c("cube", "diabetes")) |>
    ungroup() |>
    # Only three methods
    filter(method %in% c("OL-MFRL", "DIME")) |>
    # Only external predictions
    filter(prediction_type == "external") |>
    select(-prediction_type) |>
    # Only accuracy
    filter(metric_type == "accuracy") |>
    select(-metric_type) |>
    rename(
      avg_accuracy = avg_metric,
      sd_accuracy = sd_metric
    ) |>
    # No features chosen
    select(
      -c(mean_avg_features_chosen, sd_avg_features_chosen)
    ) |>
    # Only 3 cost parameters per method/dataset
    slice_head(n = 3, by = c(method, dataset)) |>
    # Visual stuff: fewer decimals
    mutate(
      cost_parameter = signif(cost_parameter, 3),
      avg_accuracy = round(avg_accuracy, 3),
      sd_accuracy = round(sd_accuracy, 3),
      # mean_avg_features_chosen = round(mean_avg_features_chosen, 1),
      # sd_avg_features_chosen = round(sd_avg_features_chosen, 2)
      mean_avg_acquisition_cost = round(mean_avg_acquisition_cost, 1),
      sd_avg_acquisition_cost = round(sd_avg_acquisition_cost, 2)
    ) |>
    # Sort by dataset first, then method
    relocate(dataset, method) |>
    arrange(dataset, method)
  # No sd columns
  # select(-c(sd_accuracy, sd_avg_features_chosen))
}

get_soft_budget_plot <- function(df, classifier_type) {
  # Filter to only include the specific classifier type
  df <- df |>
    filter(prediction_type == classifier_type) |>
    select(-prediction_type)

  # Filter out datasets that only have a single method
  df <- df |>
    group_by(dataset) |>
    filter(n_distinct(method) > 1) |>
    ungroup()

  ggplot(df, aes(
    x = mean_avg_acquisition_cost,
    y = avg_metric,
    color = method
  )) +
    geom_point() +
    geom_line() +
    geom_errorbar(
      aes(
        ymin = avg_metric - sd_metric,
        ymax = avg_metric + sd_metric
      ),
      width = 0
    ) +
    geom_errorbarh(
      aes(
        xmin = mean_avg_acquisition_cost - sd_avg_acquisition_cost,
        xmax = mean_avg_acquisition_cost + sd_avg_acquisition_cost
      ),
      height = 0
    ) +
    facet_wrap(vars(dataset), scales = "free") +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title = paste0("Metric vs avg. acquisition cost (", classifier_type, ")"),
      x = "Avg. acquisition cost",
      y = "Metric",
      color = "Method"
    ) +
    theme_bw()
}

get_params_plot <- function(df) {
  ggplot(
    df,
    aes(x = mean_avg_features_chosen, y = cost_parameter)
  ) +
    geom_point() +
    facet_wrap(vars(method, dataset), scales = "free")
}

# Which datasets we want to plot F1 for instead of accuracy
f1_datasets <- c("physionet")

results_path <- args[1]
plot_path1 <- args[2]
plot_path2 <- args[3]
plot_path3 <- args[4]

df <- read_csv(results_path, col_types = list(
  features_chosen = col_integer(),
  predicted_label_external = col_integer(),
  true_label = col_integer(),
  predicted_label_builtin = col_integer(),
  acquisition_cost = col_double(),
  method = col_factor(),
  training_seed = col_integer(),
  cost_parameter = col_double(),
  dataset = col_factor(),
  dataset_split = col_integer()
))
verify_df(df)
tidied_df <- tidy_df(df)
summarized_df <- summarize_df(tidied_df)
soft_budget_plot_builtin <- get_soft_budget_plot(summarized_df, "builtin")
soft_budget_plot_external <- get_soft_budget_plot(summarized_df, "external")
params_plot <- get_params_plot(summarized_df)

# Save the plots
ggsave(plot_path1, soft_budget_plot_builtin, width = 10, height = 6, dpi = 300)
ggsave(plot_path2, soft_budget_plot_external, width = 10, height = 6, dpi = 300)
ggsave(plot_path3, params_plot, width = 10, height = 6, dpi = 300)

if (interactive()) {
  # Markdown table to show concise results in text form
  uncluttured_df <- unclutter_df(summarized_df)
  md_table <- uncluttured_df |>
    kable(
      format = "markdown",
      col.names = c(
        "Dataset",
        "Method",
        "Cost parameter",
        "Avg. accuracy",
        "Std. accuracy",
        # "Mean avg. features chosen",
        # "Std avg. features chosen"
        "Mean avg. acquisition cost",
        "Std avg. acquisition cost"
      )
    )
  sum(nchar(md_table))
  compact_md_table <- c(
    md_table[1],
    gsub("\\s+", "", md_table[-1])
  )
  sum(nchar(compact_md_table))
  cat(compact_md_table, sep = "\n")
}

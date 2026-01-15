# Expected input dataframe with columns:
# - afa_method (str): Which method was evaluated. For example "shim2018" or "zannone2019".
# - dataset (str): Which dataset the method was evaluated on. For example "afa_context" or "mnist".
# - eval_time (float): How long the evaluation took in seconds.

options(device = "null")

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(yardstick)
  library(stringr)
  library(ggbeeswarm)
})

read_csv_safe <- function(path) {
  df <- read_csv(path, col_types = list(
    afa_method = col_factor(),
    dataset = col_factor(),
    eval_time = col_number()
  ))
  df
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2) {
  df <- read_csv_safe(args[1])
  output_path <- args[2]
} else {
  stop("Usage: Rscript plot.R eval_training_times.csv output_folder")
}

plot <- ggplot(df, aes(x = afa_method, y = eval_time, color = dataset)) +
  geom_beeswarm()
ggsave(str_c(output_path, "/beeswarm_plot.svg"), plot, create.dir = TRUE)

plot <- ggplot(df, aes(x = afa_method, y = eval_time)) +
  geom_violin()
ggsave(str_c(output_path, "/violin_plot.svg"), plot, create.dir = TRUE)

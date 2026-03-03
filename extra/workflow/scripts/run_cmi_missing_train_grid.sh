#!/usr/bin/env bash
# Run the CMI missing-train grid: 10 train_initializer values x 3 methods x 4 datasets.
#
# Usage:
#   bash extra/workflow/scripts/run_cmi_missing_train_grid.sh [JOBS]
#
# Arguments:
#   JOBS  Number of parallel Snakemake jobs (default: 4)

set -euo pipefail

JOBS="${1:-4}"

TRAIN_INITIALIZERS=(
  cold
  mcar_p01
  mcar_p03
  mcar_p05
  mar_p01
  mar_p03
  mar_p05
  mnar_logistic_p01
  mnar_logistic_p03
  mnar_logistic_p05
)

CONFIGFILES=(
  extra/workflow/conf/cmi_missing_train_all_observed_eval.yaml
  extra/workflow/conf/method_options.yaml
  extra/workflow/conf/pretrain_mapping.yaml
  extra/workflow/conf/classifier_names.yaml
)

CONFIGFILE_ARGS=""
for cf in "${CONFIGFILES[@]}"; do
  CONFIGFILE_ARGS+=" $cf"
done

echo "=== CMI Missing-Train Grid ==="
echo "Train initializers: ${TRAIN_INITIALIZERS[*]}"
echo "Parallel jobs: $JOBS"
echo ""

for init in "${TRAIN_INITIALIZERS[@]}"; do
  echo "--- Running train_initializer=$init ---"
  snakemake -s extra/workflow/snakefiles/orchestration/pipeline.smk \
    --configfile $CONFIGFILE_ARGS \
    --config "train_initializer=$init" \
    -j "$JOBS" \
    all
  echo "--- Finished train_initializer=$init ---"
  echo ""
done

echo "=== All runs complete ==="

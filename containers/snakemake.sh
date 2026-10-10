#!/bin/bash
# Run Snakemake on the login node of a site whose jobs run in an image.
#
#   containers/snakemake.sh --profile workflow/profiles/pipeline/kdd26 \
#       --workflow-profile workflow/profiles/site/arrhenius -n -p all
#
# Uses the orchestration environment containers/build.sbatch built for this
# node's architecture (see containers/bin/python). Run it from the checkout
# root, like any Snakemake invocation of the pipeline.

set -euo pipefail

exec "$(dirname "${BASH_SOURCE[0]}")/bin/python" -m snakemake "$@"

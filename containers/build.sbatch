#!/bin/bash
# Build containers/afabench.def into the image for this node's architecture.
#
# Writes <image_dir>/afabench-<arch>.sif, where <arch> is `uname -m`, so on a
# cluster whose CPU and GPU nodes differ (Arrhenius: x86_64 and aarch64) submit
# it once to each. Apptainer's cache and temporary directories live on
# node-local storage and are removed on exit; only the finished image is
# written to <image_dir>. The account and partition differ per architecture, so
# they are given at submit time. Submit from the checkout root:
#
#   sbatch -A <cpu-account> -p cpu --gpus 0 --output=<image_dir>/build-%j.log \
#     containers/build.sbatch <image_dir>
#   sbatch -A <gpu-account> -p gpu --gpus 1 --output=<image_dir>/build-%j.log \
#     containers/build.sbatch <image_dir>

#SBATCH --job-name=afabench-image
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

image_dir="$(realpath -m "${1:?image directory required}")"
project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
image="afabench-$(uname -m).sif"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/afabench-image.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
export APPTAINER_CACHEDIR="$scratch/cache" APPTAINER_TMPDIR="$scratch/tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$image_dir"

echo "host=$(hostname) arch=$(uname -m) scratch=$scratch"
cd "$project_dir"
apptainer build --fakeroot "$scratch/$image" containers/afabench.def

# Copy beside the destination, then rename, so a job reading the previous image
# never sees a half-written one.
cp "$scratch/$image" "$image_dir/.$image.partial"
mv "$image_dir/.$image.partial" "$image_dir/$image"
ls -l "$image_dir/$image"

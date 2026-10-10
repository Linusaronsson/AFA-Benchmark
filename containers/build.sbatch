#!/bin/bash
# Build containers/afabench.def into the image for this node's architecture,
# and the host environment Snakemake runs in beside it.
#
# Writes <image_dir>/afabench-<arch>.sif, where <arch> is `uname -m`, the
# uv.lock it was built from as afabench-<arch>.sif.uv.lock, and
# <image_dir>/orchestration-<arch>-<lock hash>/, the orchestration dependency
# group with its own Python, which containers/bin/python runs (ADR 0008). On
# a cluster whose CPU and GPU nodes differ (Arrhenius: x86_64 and aarch64)
# submit it once to each. Apptainer's cache and temporary directories live on
# node-local storage and are removed on exit; only the finished image, its
# lock and the environment are written to <image_dir>. The account and
# partition differ per architecture, so they are given at submit time;
# containers/submit_builds.py submits it with those of each allocation of a
# site profile that names an image:
#
#   python3 containers/submit_builds.py arrhenius
#
# which, for Arrhenius, submits from the checkout root:
#
#   sbatch -A <cpu-account> -p cpu --output=containers/build-%j.log \
#     containers/build.sbatch containers
#   sbatch -A <gpu-account> -p gpu --gpus=1 --output=containers/build-%j.log \
#     containers/build.sbatch containers
#
# containers/ is the <image_dir> containers/run.sh and containers/bin/python
# look in unless $AFABENCH_IMAGE_DIR is set.

#SBATCH --job-name=afabench-image
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

if [[ $# -ne 1 ]]; then
    cat >&2 <<'USAGE'
Usage: submit with sbatch from the checkout root, once per node architecture,
with the accounts and partitions of a site profile:

  python3 containers/submit_builds.py <site>      (or: just build-images <site>)

or by hand:

  sbatch -A <cpu-account> -p cpu --output=containers/build-%j.log \
      containers/build.sbatch containers
  sbatch -A <gpu-account> -p gpu --gpus=1 --output=containers/build-%j.log \
      containers/build.sbatch containers

See "Run jobs in an image" in docs/how-to/slurm_integration.md.
USAGE
    exit 2
fi
image_dir="$(realpath -m "$1")"
project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
arch="$(uname -m)"
image="afabench-$arch.sif"

scratch="$(mktemp -d "${TMPDIR:-/tmp}/afabench-image.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
export APPTAINER_CACHEDIR="$scratch/cache" APPTAINER_TMPDIR="$scratch/tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$image_dir"

echo "host=$(hostname) arch=$(uname -m) scratch=$scratch"
cd "$project_dir"
# The lock the image is built from, recorded beside it, because the pipeline
# checks it on nodes that cannot run an image of this architecture.
cp uv.lock "$scratch/uv.lock"
apptainer build --fakeroot "$scratch/$image" containers/afabench.def

# Copy beside the destination, then rename, so a job reading the previous image
# never sees a half-written one.
cp "$scratch/$image" "$image_dir/.$image.partial"
mv "$image_dir/.$image.partial" "$image_dir/$image"
cp "$scratch/uv.lock" "$image_dir/.$image.uv.lock.partial"
mv "$image_dir/.$image.uv.lock.partial" "$image_dir/$image.uv.lock"
ls -l "$image_dir/$image" "$image_dir/$image.uv.lock"

# The orchestration environment, with the uv the image was built with, whose
# binary is static. A new uv.lock gets a new directory, so building for it
# leaves the environment of running jobs alone until it is complete; only
# then are older ones removed. Rebuilding for the same uv.lock replaces it
# in place.
lock_hash="$(sha256sum "$scratch/uv.lock" | cut -c 1-16)"
environment="$image_dir/orchestration-$arch-$lock_hash"
apptainer exec "$image_dir/$image" cat /usr/local/bin/uv > "$scratch/uv"
chmod +x "$scratch/uv"
rm -rf "$environment"
UV_PROJECT_ENVIRONMENT="$environment/venv" \
    UV_PYTHON_INSTALL_DIR="$environment/python" \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_CACHE_DIR="$scratch/uv-cache" UV_LINK_MODE=copy \
    "$scratch/uv" sync --frozen --only-group orchestration --no-install-project
for old in "$image_dir/orchestration-$arch-"*; do
    if [[ "$old" != "$environment" ]]; then
        rm -rf "$old"
    fi
done
"$environment/venv/bin/python" -m snakemake --version

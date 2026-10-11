#!/bin/bash
#SBATCH --job-name=l2m-timing
#SBATCH --partition=gpu
#SBATCH --account=naiss2026-4-1737-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1400M
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=experiments/l2m_compute_probe/runs/slurm-%A_%a.log

# Production-shape timing probe, NOT a benchmark-quality fitted method.
# Each array task owns its outputs; no production artifacts are written.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source_name=real
if [[ "$SLURM_ARRAY_TASK_ID" = 1 ]]; then
    source_name=synthetic
fi
root="experiments/l2m_compute_probe/runs/${PROBE_REUSE_JOB:-$SLURM_ARRAY_JOB_ID}_${source_name}"
mkdir -p "$root"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export WANDB_MODE=disabled

git rev-parse HEAD > "$root/commit.txt"
nvidia-smi -q > "$root/gpu-info.txt"
lscpu > "$root/cpu-info.txt"
printf 'host=%s\nCUDA_VISIBLE_DEVICES=%s\nSLURM_JOB_GPUS=%s\n' \
    "$HOSTNAME" "${CUDA_VISIBLE_DEVICES:-}" "${SLURM_JOB_GPUS:-}" > "$root/allocation-${SLURM_ARRAY_JOB_ID}.txt"
# Arrhenius device isolation renumbers the allocated GPU to index 0;
# SLURM_JOB_GPUS is the physical index and is not an NVML index here.
nvidia-smi --query-gpu=uuid --format=csv,noheader > "$root/allocated-gpu-${SLURM_ARRAY_JOB_ID}.txt"
test "$(wc -l < "$root/allocated-gpu-${SLURM_ARRAY_JOB_ID}.txt")" -eq 1
nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv --loop=1 > "$root/gpu-samples-${SLURM_ARRAY_JOB_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
if [[ ! -f "$root/stages.tsv" ]]; then
    printf 'stage\tstart_utc\tend_utc\texit_status\n' > "$root/stages.tsv"
fi

run_stage() {
    local stage="$1"
    shift
    local start status=0
    start=$(date -u +%FT%TZ)
    /usr/bin/time -v -o "$root/$stage.time.txt" containers/run.sh "$@" > "$root/$stage.log" 2>&1 || status=$?
    printf '%s\t%s\t%s\t%s\n' "$stage" "$start" "$(date -u +%FT%TZ)" "$status" >> "$root/stages.tsv"
    if [[ "$status" != 0 ]]; then
        tail -n 60 "$root/$stage.log"
        return "$status"
    fi
}

if [[ -z "${PROBE_REUSE_JOB:-}" ]]; then
run_stage dataset python scripts/dataset_generation/generate_dataset.py \
    dataset=miniboone 'dataset_realization_indices=[0]' 'seeds=[0]' \
    "save_path=$root/datasets" "hydra.run.dir=$root/hydra/dataset"

# L2M does not use the external classifier during fitting. One epoch makes
# a valid prerequisite bundle and permits throughput-only evaluation.
run_stage classifier python scripts/train_classifier/masked_mlp_classifier.py \
    experiment@_global_=miniboone initializer=cold unmasker=direct device=cuda seed=0 \
    "train_dataset_path=$root/datasets/0/train.bundle" \
    "val_dataset_path=$root/datasets/0/val.bundle" \
    "save_path=$root/classifier.bundle" epochs=1 use_wandb=false smoke_test=false \
    "hydra.run.dir=$root/hydra/classifier"
fi

common=(initializer=cold unmasker=direct dataset_key=miniboone device=cuda seed=0
    use_wandb=false smoke_test=false "feature_source=$source_name"
    "train_dataset_bundle_path=$root/datasets/0/train.bundle"
    "val_dataset_bundle_path=$root/datasets/0/val.bundle"
    "classifier_bundle_path=$root/classifier.bundle")

# Keep production architecture, batch_size=8, sequence_length=1000,
# validation tasks=64 and checkpoint intervals=500/200 unchanged.
if [[ -z "${PROBE_REUSE_JOB:-}" ]]; then
    run_stage pretrain python scripts/pretrain_model/l2m.py "${common[@]}" \
        n_steps=1000 "save_path=$root/pretrained.bundle" \
        "hydra.run.dir=$root/hydra/pretrain"
else
    test -d "$root/pretrained.bundle"
fi
run_stage train python scripts/train_method/l2m.py "${common[@]}" \
    n_steps=1000 hard_budget=15 soft_budget_param=null "method_name=l2m_${source_name}_feature_prior" \
    "pretrained_model_bundle_path=$root/pretrained.bundle" \
    "save_path=$root/method.bundle" "hydra.run.dir=$root/hydra/train"

# Production context_set_size=900 and eval batch_size=128. Predictions are
# not benchmark results: both the method and classifier are undertrained.
for budget in 5 10 15; do
    run_stage "eval_$budget" python scripts/eval/eval_afa_method.py \
        "method_bundle_path=$root/method.bundle" initializer=cold unmasker=direct \
        "dataset_bundle_path=$root/datasets/0/test.bundle" \
        "classifier_bundle_path=$root/classifier.bundle" \
        "save_path=$root/eval_$budget.parquet" seed=0 device=cuda \
        "hard_budget=$budget" batch_size=128 eval_only_n_samples=2048 \
        use_wandb=false smoke_test=false "hydra.run.dir=$root/hydra/eval_$budget"
done
printf 'Completed timing probe: %s\n' "$root"

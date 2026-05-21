# Format, lint, and type-check
check:
    uv run ruff format .
    uv run ruff check . --fix
    uv run basedpyright --warnings

# Fast tests
test:
    uv run pytest .

# Run all tests including expensive optional tests
test-full:
    uv run pytest . -m "optional or not optional"

# QA = static checks + fast tests
qa:
    just check
    just test

local-bundle-check:
    uv run scripts/hub/check_local.py \
        --configfile extra/workflow/conf/method_options.yaml \
        --configfile extra/workflow/conf/pretrain_mapping.yaml \
        --configfile extra/workflow/conf/classifier_names.yaml \
        --configfile extra/workflow/conf/eval_hard_budgets.yaml \
        --configfile extra/workflow/conf/soft_budget_params.yaml \
        --configfile extra/workflow/conf/unmaskers.yaml \
        --configfile extra/workflow/conf/datasets_main.yaml \
        --configfile extra/workflow/conf/methods.yaml

local-cpu-run:
    WANDB_PROJECT=afabench uv run snakemake \
        -s extra/workflow/snakefiles/orchestration/pipeline.smk \
        all \
        --configfile \
          extra/workflow/conf/eval_hard_budgets.yaml \
          extra/workflow/conf/methods.yaml \
          extra/workflow/conf/method_sets.yaml \
          extra/workflow/conf/method_options.yaml \
          extra/workflow/conf/pretrain_mapping.yaml \
          extra/workflow/conf/soft_budget_params.yaml \
          extra/workflow/conf/unmaskers.yaml \
          extra/workflow/conf/classifier_names.yaml \
          extra/workflow/conf/datasets_main.yaml \
        --config \
          eval_dataset_split=test \
          "dataset_instance_indices=[0,1,2,3,4]" \
          smoke_test=false \
          use_wandb=true \
          device=cpu \
        --jobs 8

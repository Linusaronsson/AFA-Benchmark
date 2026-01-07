# Show available commands
list:
    @just --list

# Run all the formatting, linting, and testing commands
qa:
    # Format
    uv run ruff format .

    # Linting
    uv run ruff check . --fix

    # LSP checks
    pre-commit run basedpyright --all-files

    # Testing
    uv run pytest . --tb=no

# Run coverage, and build to HTML
coverage:
    uv run coverage run -m pytest .
    uv run coverage report -m
    uv run coverage html

# --- Pipeline ---
rl_and_dummy_all_smoke:
    snakemake -s extra/workflow/snakefiles/RL_and_dummy.smk --configfile extra/workflow/conf/{datasets,hard_budgets,methods,method_options,soft_budget_params,unmaskers}.yaml --config smoke_test=true use_wandb=true --forceall --jobs 10 --rerun-incomplete

# --- Pretraining RL methods ---

pretrain_shim2018_cube_without_noise *extra_args='':
    uv run scripts/pretrain/shim2018.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle/ \
                    save_path=tmp/shim2018_pretrained_cube_without_noise.bundle \
                    device=cpu \
                    seed=null \
                    use_wandb=true \
                    experiment@_global_=cube


pretrain_shim2018_afa_context_without_noise *extra_args='':
    uv run scripts/pretrain/shim2018.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/afa_context_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/afa_context_without_noise/0/val.bundle/ \
                    save_path=tmp/shim2018_pretrained_afa_context_without_noise.bundle \
                    device=cuda \
                    seed=null \
                    use_wandb=true \
                    experiment@_global_=afa_context

pretrain_shim2018_synthetic_mnist_without_noise *extra_args='':
    uv run scripts/pretrain/shim2018.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/val.bundle/ \
                    save_path=tmp/shim2018_pretrained_synthetic_mnist_without_noise.bundle \
                    device=cuda \
                    seed=null \
                    use_wandb=true \
                    experiment@_global_=synthetic_mnist

pretrain_zannone2019_cube_without_noise *extra_args='':
    uv run scripts/pretrain/zannone2019.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle/ \
                    save_path=tmp/zannone2019_pretrained_cube_without_noise.bundle \
                    device=cpu \
                    seed=null \
                    use_wandb=true \
                    experiment@_global_=cube

pretrain_zannone2019_synthetic_mnist_without_noise *extra_args='':
    uv run scripts/pretrain/zannone2019.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/val.bundle/ \
                    save_path=tmp/zannone2019_pretrained_synthetic_mnist_without_noise.bundle \
                    device=cuda \
                    seed=null \
                    use_wandb=true \
                    experiment@_global_=synthetic_mnist

pretrain_kachuee2019_synthetic_mnist_without_noise *extra_args='':
    uv run scripts/pretrain/kachuee2019.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/val.bundle/ \
                    save_path=tmp/kachuee2019_pretrained_synthetic_mnist_without_noise.bundle \
                    device=cuda \
                    seed=null \
                    use_wandb=true \
                    experiment@_global_=synthetic_mnist

pretrain_kachuee2019_cube_without_noise *extra_args='':
    uv run scripts/pretrain/kachuee2019.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle/ \
                    save_path=tmp/kachuee2019_pretrained_cube_without_noise.bundle \
                    device=cpu \
                    seed=null \
                    use_wandb=true \
                    experiment@_global_=cube

# --- Train RL methods, hard budget ---


train_shim2018_cube_without_noise_hard *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_cube_without_noise.bundle \
        save_path=tmp/shim2018_trained_cube_without_noise_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=direct \
        mdp.hard_budget=5 \
        soft_budget_param=null \
        device=cpu \
        seed=null \
        use_wandb=true \
        experiment@_global_=cube \
        activate_joint_training_after_fraction=0.0 \
        rl_training_loop.n_batches=2000 \
        rl_training_loop.eval_n_times=20 \
        agent.eps_init=0.5 \
        agent.eps_end=0.0 \
        agent.eps_annealing_fraction=0.8 \
        agent.lr=1e-3

train_shim2018_synthetic_mnist_without_noise_hard *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_synthetic_mnist_without_noise.bundle \
        save_path=tmp/shim2018_trained_synthetic_mnist_without_noise_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=28x28_to_7x7 \
        mdp.hard_budget=5 \
        soft_budget_param=null \
        device=cpu \
        seed=null \
        use_wandb=true \
        experiment@_global_=synthetic_mnist \
        activate_joint_training_after_fraction=0.0 \
        rl_training_loop.n_batches=2000 \
        rl_training_loop.eval_n_times=20 \
        agent.eps_init=0.5 \
        agent.eps_end=0.0 \
        agent.eps_annealing_fraction=0.8 \
        agent.lr=1e-3

train_zannone2019_cube_without_noise_hard *extra_args='':
    uv run scripts/train/zannone2019.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/zannone2019_pretrained_cube_without_noise.bundle \
        save_path=tmp/zannone2019_trained_cube_without_noise_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=direct \
        mdp.hard_budget=5 \
        soft_budget_param=null \
        device=cpu \
        seed=null \
        use_wandb=true \
        experiment@_global_=cube \
        n_generated_samples=0 \
        rl_training_loop.n_batches=2000 \
        rl_training_loop.eval_n_times=20 \
        agent.lr=1e-3

train_zannone2019_synthetic_mnist_without_noise_hard *extra_args='':
    uv run scripts/train/zannone2019.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/zannone2019_pretrained_synthetic_mnist_without_noise.bundle \
        save_path=tmp/zannone2019_trained_synthetic_mnist_without_noise_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=28x28_to_7x7 \
        mdp.hard_budget=5 \
        soft_budget_param=null \
        device=cpu \
        seed=null \
        use_wandb=true \
        experiment@_global_=synthetic_mnist \
        n_generated_samples=0 \
        rl_training_loop.n_batches=2000 \
        rl_training_loop.eval_n_times=20 \

# --- Train RL methods, soft budget ---

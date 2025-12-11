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

# Build the project, useful for checking that packaging is correct
# build:
#     rm -rf build
#     rm -rf dist
#     uv build

pretrain_shim2018_cube_without_noise *extra_args='':
    uv run scripts/pretrain/shim2018.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle/ \
                    save_path=tmp/shim2018_pretrained_cube_without_noise.bundle \
                    device=cpu \
                    seed=42 \
                    use_wandb=true \
                    +experiment@_global_=cube

train_shim2018_cube_without_noise_hard *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_cube_without_noise.bundle \
        save_path=tmp/shim2018_trained_cube_without_noise_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=direct \
        hard_budget=5 \
        soft_budget_param=null \
        device=cpu \
        seed=42 \
        use_wandb=true \
        +experiment@_global_=cube \
        activate_joint_training_after_fraction=0.0 \
        n_batches=2000 \
        agent.eps_init=0.5 \
        agent.eps_end=0.0 \
        agent.eps_annealing_fraction=0.8 \
        agent.lr=1e-3 \
        eval_n_times=20
# batch_size=8192

train_shim2018_cube_without_noise_soft *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/cube_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_cube_without_noise.bundle \
        save_path=tmp/shim2018_trained_cube_without_noise_soft.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=direct \
        hard_budget=null \
        soft_budget_param=0.01 \
        device=cpu \
        seed=42 \
        use_wandb=true \
        +experiment@_global_=cube \
        activate_joint_training_after_fraction=1.0 \
        n_batches=2000 \
        agent.eps_init=0.5 \
        agent.eps_end=0.0 \
        agent.eps_annealing_fraction=0.8 \
        agent.lr=1e-3 \
        eval_n_times=20

pretrain_shim2018_afa_context_without_noise *extra_args='':
    uv run scripts/pretrain/shim2018.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/afa_context_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/afa_context_without_noise/0/val.bundle/ \
                    save_path=tmp/shim2018_pretrained_afa_context_without_noise.bundle \
                    device=cpu \
                    seed=42 \
                    use_wandb=true \
                    +experiment@_global_=afa_context
train_shim2018_afa_context_without_noise_hard *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/afa_context_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/afa_context_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_afa_context_without_noise.bundle \
        save_path=tmp/shim2018_trained_afa_context_without_noise_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=direct \
        hard_budget=5 \
        soft_budget_param=null \
        device=cpu \
        seed=42 \
        use_wandb=true \
        +experiment@_global_=afa_context \
        activate_joint_training_after_fraction=1.0 \
        n_batches=2000 \
        agent.eps_init=0.5 \
        agent.eps_end=0.0 \
        agent.eps_annealing_fraction=0.8 \
        agent.lr=1e-3 \
        eval_n_times=20

pretrain_shim2018_synthetic_mnist_without_noise *extra_args='':
    uv run scripts/pretrain/shim2018.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/val.bundle/ \
                    save_path=tmp/shim2018_pretrained_synthetic_mnist_without_noise.bundle \
                    device=cuda \
                    seed=42 \
                    use_wandb=true \
                    +experiment@_global_=synthetic_mnist
train_shim2018_synthetic_mnist_without_noise_hard *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/synthetic_mnist_without_noise/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_synthetic_mnist_without_noise.bundle \
        save_path=tmp/shim2018_trained_synthetic_mnist_without_noise_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=28x28_to_7x7 \
        hard_budget=1 \
        soft_budget_param=null \
        device=cpu \
        seed=42 \
        use_wandb=true \
        +experiment@_global_=synthetic_mnist \
        activate_joint_training_after_fraction=1.0 \
        n_batches=2000 \
        agent.eps_init=0.5 \
        agent.eps_end=0.0 \
        agent.eps_annealing_fraction=0.8 \
        agent.lr=1e-3 \
        eval_n_times=20

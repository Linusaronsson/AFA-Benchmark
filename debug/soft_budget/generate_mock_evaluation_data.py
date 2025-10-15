import numpy as np
import pandas as pd


def generate_mock_data(
    n_methods=5,
    n_seeds=5,
    n_cost_params=5,
    n_datasets=5,
    n_samples=200,
    n_classes=4,
    n_splits=3,
    random_state=42,
):
    np.random.seed(random_state)
    methods = [f"method_{i + 1}" for i in range(n_methods)]
    seeds = [100 + i for i in range(n_seeds)]
    # Define different cost_param ranges for each method
    method_cost_param_ranges = {
        "method_1": np.linspace(0.01, 0.1, n_cost_params),
        "method_2": np.linspace(1, 5, n_cost_params),
        "method_3": np.linspace(0.5, 2, n_cost_params),
        "method_4": np.linspace(10, 100, n_cost_params),
        "method_5": np.linspace(0.001, 0.01, n_cost_params),
    }
    datasets = [f"dataset_{chr(65 + i)}" for i in range(n_datasets)]
    # Assign a different max_features (x) for each dataset
    dataset_max_features = {
        dataset: np.random.randint(5, 25) for dataset in datasets
    }

    # For each dataset, for each split, shuffle and assign split-local indices
    dataset_split_sample_map = {}
    dataset_split_true_labels = {}
    rng = np.random.RandomState(random_state)
    for dataset in datasets:
        # Generate true labels for all samples in this dataset
        true_labels = rng.randint(0, n_classes, size=n_samples)
        for split in range(n_splits):
            # Shuffle indices for this split
            split_rng = np.random.RandomState(
                random_state + split * 1000 + hash(dataset) % 1000
            )
            indices = np.arange(n_samples)
            split_rng.shuffle(indices)
            # Map split-local index to global sample index
            dataset_split_sample_map[(dataset, split)] = indices.copy()
            # Store true labels for this split (global index)
            dataset_split_true_labels[(dataset, split)] = true_labels[indices]
    # 2. Generate results file: method, training_seed, cost_parameter, dataset, dataset_split, sample, features_chosen, predicted_label_builtin, predicted_label_external
    results_rows = []

    # Method-dependent offsets for features and accuracy
    method_offsets = {
        "method_1": {"features": 0.0, "accuracy": 0.0},
        "method_2": {"features": 0.2, "accuracy": 0.1},
        "method_3": {"features": -0.2, "accuracy": -0.1},
        "method_4": {"features": 0.3, "accuracy": 0.15},
        "method_5": {"features": -0.3, "accuracy": -0.15},
    }

    for method in methods:
        # Use method-specific cost_param range
        cost_params = method_cost_param_ranges.get(
            method, np.linspace(0.1, 1, n_cost_params)
        )
        for seed in seeds:
            # Set a unique random seed for each method/seed combination
            np.random.seed(seed * 1000 + hash(method) % 1000)
            # Seed-dependent offsets for features and accuracy
            seed_offset_features = np.random.uniform(-0.2, 0.2)
            seed_offset_accuracy = np.random.uniform(-0.1, 0.1)
            # Method-dependent offsets for features and accuracy
            method_offset_features = method_offsets[method]["features"]
            method_offset_accuracy = method_offsets[method]["accuracy"]
            for cost_param in cost_params:
                for dataset in datasets:
                    max_features = dataset_max_features[dataset]
                    for split in range(n_splits):
                        indices = dataset_split_sample_map[(dataset, split)]
                        split_true_labels = dataset_split_true_labels[
                            (dataset, split)
                        ]
                        for split_sample, global_sample in enumerate(indices):
                            cost_param_range = np.ptp(cost_params)
                            if cost_param_range == 0:
                                norm_cost = 0
                            else:
                                norm_cost = (
                                    cost_param - np.min(cost_params)
                                ) / cost_param_range
                            # Add both method and seed offsets to mean_features
                            mean_features = (
                                norm_cost
                                + seed_offset_features
                                + method_offset_features
                            ) * max_features
                            features_chosen = int(
                                np.clip(
                                    np.random.normal(
                                        loc=mean_features,
                                        scale=max(1, 0.15 * max_features),
                                    ),
                                    0,
                                    max_features,
                                )
                            )
                            # Acquisition cost: proportional to features_chosen, with small noise
                            cost_per_feature = 1.0  # Can be customized per dataset/method if desired
                            acquisition_cost = (
                                features_chosen * cost_per_feature
                                + np.random.normal(0, 0.1)
                            )
                            acquisition_cost = max(acquisition_cost, 0.0)
                            methods_with_builtin = {"method_1", "method_3"}
                            has_builtin = method in methods_with_builtin

                            true_label = split_true_labels[split_sample]

                            # Add both method and seed offsets to accuracy
                            prob_correct_external = min(
                                0.5
                                + 0.5 * (features_chosen / max_features)
                                + seed_offset_accuracy
                                + method_offset_accuracy,
                                0.98,
                            )
                            prob_correct_external = max(
                                prob_correct_external, 0.01
                            )
                            prob_correct_builtin = min(
                                0.45
                                + 0.5 * (features_chosen / max_features)
                                + seed_offset_accuracy
                                + method_offset_accuracy,
                                0.95,
                            )
                            prob_correct_builtin = max(
                                prob_correct_builtin, 0.01
                            )

                            if np.random.rand() < prob_correct_external:
                                pred_label_external = true_label
                            else:
                                pred_label_external = np.random.choice(
                                    [
                                        l
                                        for l in range(n_classes)
                                        if l != true_label
                                    ]
                                )

                            if has_builtin:
                                if np.random.rand() < prob_correct_builtin:
                                    pred_label_builtin = true_label
                                else:
                                    pred_label_builtin = np.random.choice(
                                        [
                                            l
                                            for l in range(n_classes)
                                            if l != true_label
                                        ]
                                    )
                            else:
                                pred_label_builtin = None

                            results_rows.append(
                                {
                                    "method": method,
                                    "training_seed": seed,
                                    "cost_parameter": float(cost_param),
                                    "dataset": dataset,
                                    "dataset_split": split,
                                    "features_chosen": features_chosen,
                                    "acquisition_cost": acquisition_cost,
                                    "predicted_label_builtin": pred_label_builtin,
                                    "predicted_label_external": pred_label_external,
                                    "true_label": true_label,
                                }
                            )
    results_df = pd.DataFrame(results_rows)
    return results_df


if __name__ == "__main__":
    results_df = generate_mock_data()

    results_df["training_seed"] = results_df["training_seed"].astype(int)
    results_df["features_chosen"] = results_df["features_chosen"].astype(int)
    results_df["predicted_label_external"] = results_df[
        "predicted_label_external"
    ].astype(int)
    # Use pandas nullable Int64 for columns that can be NA/None
    results_df["predicted_label_builtin"] = results_df[
        "predicted_label_builtin"
    ].astype("Int64")

    results_df.to_csv("eval_results.csv", index=False)
    print("Mock results saved to eval_results.csv")

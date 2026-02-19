import pandas as pd
import torch


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_missingness_reliance(
    eval_df: pd.DataFrame,
    initial_missing_mask: torch.BoolTensor,
    _n_selection_choices: int,
) -> dict[str, float]:
    """
    Compute missingness reliance metrics from AFA evaluation results.

    Missingness reliance measures how much an AFA policy depends on features
    that were initially missing. This directly corresponds to the rho metric
    from Stempfle et al., adapted for the AFA setting.

    In Stempfle et al., rho(h, x) := max_j 1[a_h(X,j)=1 AND x_j=na].
    In the AFA setting, we measure: for each instance, did the policy acquire
    any feature that was marked as missing in the initial mask?

    Args:
        eval_df: DataFrame from eval_afa_method(), containing columns
            'prev_selections_performed', 'action_performed', 'idx'.
        initial_missing_mask: Boolean tensor of shape (n_samples, n_features)
            where True means the feature was initially missing (malearn
            convention). This is the INVERSE of the AFA initial feature mask.
        n_selection_choices: Number of possible selections (excluding stop).

    Returns:
        Dictionary with:
            - 'missingness_reliance': Fraction of instances where the policy
              acquired at least one initially-missing feature.
            - 'mean_missing_acquisitions': Mean number of initially-missing
              features acquired per instance.
            - 'mean_acquisition_fraction_missing': Mean fraction of acquired
              features that were initially missing, per instance.
    """
    n_samples = initial_missing_mask.shape[0]

    # Extract final selections per sample (all non-stop actions performed)
    sample_selections: dict[int, list[int]] = {}
    for _, row in eval_df.iterrows():
        idx = int(row["idx"])
        action = int(row["action_performed"])
        if action > 0:  # Not a stop action
            selection_idx = action - 1
            if idx not in sample_selections:
                sample_selections[idx] = []
            sample_selections[idx].append(selection_idx)

    relied_on_missing = 0
    total_missing_acquisitions = 0
    total_acquisitions = 0
    fraction_missing_per_sample = []

    for sample_idx in range(n_samples):
        selections = sample_selections.get(sample_idx, [])
        if not selections:
            fraction_missing_per_sample.append(0.0)
            continue

        total_acquisitions += len(selections)

        # Check which acquired features were initially missing
        missing_count = 0
        for sel_idx in selections:
            if (
                sel_idx < initial_missing_mask.shape[-1]
                and initial_missing_mask[sample_idx, sel_idx].item()
            ):
                missing_count += 1

        if missing_count > 0:
            relied_on_missing += 1
        total_missing_acquisitions += missing_count
        fraction_missing_per_sample.append(
            missing_count / len(selections) if selections else 0.0
        )

    missingness_reliance = _safe_divide(relied_on_missing, n_samples)
    mean_missing_acquisitions = _safe_divide(
        total_missing_acquisitions, n_samples
    )
    mean_fraction = _safe_divide(
        sum(fraction_missing_per_sample), len(fraction_missing_per_sample)
    )

    return {
        "missingness_reliance": missingness_reliance,
        "mean_missing_acquisitions": mean_missing_acquisitions,
        "mean_acquisition_fraction_missing": mean_fraction,
    }

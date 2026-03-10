from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from afabench.common.datasets.datasets import CubeNMARDataset

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset

CUBE_NM_AR_RISK_COLUMNS = {
    "cube_nm_ar_is_risky_context",
    "cube_nm_ar_relevant_block_blocked",
    "cube_nm_ar_is_relevant_block_action",
    "cube_nm_ar_is_rescue_action",
}


def augment_cube_nm_ar_eval_df(
    df_eval: pd.DataFrame,
    dataset: AFADataset,
) -> pd.DataFrame:
    if not isinstance(dataset, CubeNMARDataset):
        return df_eval

    features, _labels = dataset.get_all_data()
    context_idx = features[:, : dataset.n_contexts].argmax(dim=1).cpu().numpy()
    admin_start = dataset.n_contexts + dataset.n_hint_features
    relevant_block_blocked = features[:, admin_start + 1].cpu().numpy() > 0.5
    is_risky_context = context_idx >= dataset.n_safe_contexts
    relevant_action_start = 2 + context_idx * dataset.block_size
    relevant_action_end = relevant_action_start + dataset.block_size - 1
    rescue_action = 2 + dataset.n_contexts * dataset.block_size

    metadata = pd.DataFrame(
        {
            "idx": range(len(dataset)),
            "cube_nm_ar_context_idx": context_idx,
            "cube_nm_ar_is_risky_context": is_risky_context,
            "cube_nm_ar_relevant_block_blocked": relevant_block_blocked,
            "cube_nm_ar_relevant_action_start": relevant_action_start,
            "cube_nm_ar_relevant_action_end": relevant_action_end,
            "cube_nm_ar_rescue_action": rescue_action,
        }
    )
    df_augmented = df_eval.merge(
        metadata, on="idx", how="left", validate="many_to_one"
    )
    df_augmented["cube_nm_ar_is_relevant_block_action"] = df_augmented[
        "action_performed"
    ].ge(df_augmented["cube_nm_ar_relevant_action_start"]) & df_augmented[
        "action_performed"
    ].le(df_augmented["cube_nm_ar_relevant_action_end"])
    df_augmented["cube_nm_ar_is_rescue_action"] = df_augmented[
        "action_performed"
    ].eq(df_augmented["cube_nm_ar_rescue_action"])
    return df_augmented


def add_episode_id(df_eval: pd.DataFrame) -> pd.DataFrame:
    starts = df_eval["prev_selections_performed"].astype(str).eq("[]")
    episode_idx = starts.groupby(df_eval["idx"]).cumsum() - 1
    return df_eval.assign(
        episode_id=df_eval["idx"].astype(str) + ":" + episode_idx.astype(str)
    )


def summarize_cube_nm_ar_episodes(
    df_eval: pd.DataFrame,
) -> pd.DataFrame | None:
    if not CUBE_NM_AR_RISK_COLUMNS.issubset(df_eval.columns):
        return None

    df_with_episode_id = add_episode_id(df_eval)
    final = (
        df_with_episode_id.groupby("episode_id", as_index=False).tail(1).copy()
    )
    first = (
        df_with_episode_id.groupby("episode_id", as_index=False).head(1).copy()
    )
    first_actions = first.loc[:, ["episode_id", "action_performed"]].copy()
    first_actions.columns = ["episode_id", "first_action"]
    action_flags = (
        pd.DataFrame(
            df_with_episode_id.groupby("episode_id").agg(
                context_acquired=(
                    "action_performed",
                    lambda x: (x == 1).any(),
                ),
                relevant_block_acquired=(
                    "cube_nm_ar_is_relevant_block_action",
                    "any",
                ),
                rescue_acquired=("cube_nm_ar_is_rescue_action", "any"),
            )
        )
        .reset_index()
        .copy()
    )
    episode_df = final.merge(
        first_actions,
        on="episode_id",
        how="left",
        validate="one_to_one",
    ).merge(action_flags, on="episode_id", how="left", validate="one_to_one")
    episode_df["correct"] = (
        episode_df["external_predicted_class"] == episode_df["true_class"]
    ).astype(float)

    risky_mask = episode_df["cube_nm_ar_is_risky_context"].astype(bool)
    blocked_mask = episode_df["cube_nm_ar_relevant_block_blocked"].astype(bool)
    informed_stop = episode_df["relevant_block_acquired"].astype(
        bool
    ) & episode_df["rescue_acquired"].astype(bool)
    episode_df["unsafe_stop"] = risky_mask & ~informed_stop
    episode_df["avoidable_unsafe_stop"] = episode_df["unsafe_stop"] & (
        risky_mask & ~blocked_mask
    )
    return episode_df

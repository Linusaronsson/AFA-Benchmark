from typing import Any, TypeVar

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from tqdm import tqdm

from common.custom_types import (
    AFASelectFn,
    AFAPredictFn,
    AFADataset,
    FeatureMask,
    Label,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F


def aggregate_metrics(prediction_history, y_true) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute accuracy, F1 and BCE across feature-selection budgets.

    If y_true contains exactly two unique classes   → average="binary"
    Otherwise                                       → average="weighted"

    Parameters
    ----------
    prediction_history : Tensor[float, shape=(n_samples, budget, n_classes)]
    y_true : Tensor[int, shape=(n_samples,)]
        Ground-truth labels, same order as prediction_history.

    Returns
    -------
    accuracy_all : Tensor[float, shape=(budget,)]
    f1_all       : Tensor[float, shape=(budget,)]
    bce_all      : Tensor[float, shape=(budget,)]
    """

    B = prediction_history.shape[1]
    # if any(len(row) != B for row in prediction_history_all):
    #     raise ValueError("All inner lists must have the same length (budget B).")

    # Decide the F1 averaging mode
    classes = np.unique(y_true)
    if len(classes) == 2:
        f1_kwargs = {"average": "binary"}  # treat the larger label as positive
    else:
        f1_kwargs = {"average": "weighted"}

    accuracy_all, f1_all, bce_all = [], [], []
    for i in range(B):
        preds_i = [torch.argmax(row) for row in prediction_history[:, i]]
        accuracy_all.append(accuracy_score(y_true, preds_i))
        f1_all.append(f1_score(y_true, preds_i, **f1_kwargs))
        probs_i = torch.stack([row for row in prediction_history[:, i]])
        bce_all.append(
            F.binary_cross_entropy(
                probs_i, F.one_hot(y_true, num_classes=probs_i.shape[-1]).float()
            )
        )

    return (
        torch.tensor(accuracy_all, dtype=torch.float64),
        torch.tensor(f1_all, dtype=torch.float64),
        torch.tensor(bce_all, dtype=torch.float64),
    )


# def evaluator(
#     feature_mask_history_all: list[list[FeatureMask]],
#     prediction_history_all: list[list[Label]],
#     labels_all: list[Label],
# ) -> dict[str, Any]:
#     assert (
#         len(feature_mask_history_all) == len(prediction_history_all) == len(labels_all)
#     ), "All three lists must have the same length"
#
#     labels_all: Tensor = torch.stack(labels_all)
#     labels_all = torch.argmax(labels_all, dim=1)
#
#     accuracy_all, f1_all, bce_all = aggregate_metrics(
#         prediction_history_all, labels_all
#     )
#
#     return {
#         "accuracy_all": accuracy_all.detach().cpu(),
#         "f1_all": f1_all.detach().cpu(),
#         "bce_all": bce_all.detach().cpu(),
#         "feature_mask_history_all": [
#             [t.detach().cpu() for t in sublist] for sublist in feature_mask_history_all
#         ],
#     }


def eval_afa_method(
    afa_select_fn: AFASelectFn,
    dataset: AFADataset,  # NOTE: should also be annotated with torch.utils.data.Dataset
    budget: int,
    afa_predict_fn: AFAPredictFn,
    only_n_samples: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Evaluate an AFA method.

    Args:
        afa_select_fn (AFASelectFn): How to select new features.
        dataset (AFADataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        budget (int): The number of features to select.
        afa_predict_fn (AFAPredictFn): The label prediction function to use for evaluation.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.
        batch_size (int): How many AFA episodes to run concurrently.

    Returns:
        dict[str, float]: A dictionary containing the evaluation results.

    """
    # TODO: make this an argument
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Store feature mask history, label prediction history, and true label for each sample in the dataset
    feature_mask_history_all = []  # list[Tensor[budget,batch_size,n_features]] (n_batches)
    prediction_history_all = []  # list[Tensor[budget,batch_size,n_classes]]] (n_batches)
    labels_all: list[Label] = [] # (n_batches)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )  # basedpyright: ignore

    # Loop over the dataset
    n_evaluated_samples = 0
    for batch in tqdm(
        dataloader,
        total=only_n_samples // batch_size if only_n_samples is not None else len(dataloader),
        desc="Evaluating",
    ):
        # Each batch has features and labels
        features, labels = batch
        # features = features.to(device)
        # labels = labels.to(device)

        # Immediately store the true labels for this batch
        labels_all.append(labels)

        # We will keep a history of which features have been observed, in case its relevant for evaluation
        feature_mask_history = torch.zeros(budget, features.shape[0], features.shape[1], dtype=torch.bool) # budget, batch_size, n_features

        # And also a history of predictions
        prediction_history = torch.zeros(budget, features.shape[0], labels.shape[1], dtype=torch.float32)  # budget, batch_size, n_classes

        # Start with all features unobserved
        feature_mask = torch.zeros_like(features, dtype=torch.bool)
        masked_features = features.clone()
        masked_features[~feature_mask] = 0.0

        # Let AFA method select features for a fixed number of steps
        for i in range(budget):
            # Always calculate a prediction
            predictions = afa_predict_fn(masked_features, feature_mask)

            prediction_history[i] = predictions

            # Select new features
            selections = afa_select_fn(masked_features, feature_mask).squeeze(-1)

            # Update the feature mask and masked features
            feature_mask[torch.arange(feature_mask.shape[0]), selections] = True
            masked_features[torch.arange(feature_mask.shape[0]), selections] = features[
                torch.arange(feature_mask.shape[0]), selections
            ]

            # Store a copy of the feature mask in history
            feature_mask_history[i] = feature_mask.clone()

        # Add the feature mask history and prediction history of this batch to the overall history
        feature_mask_history_all.append(feature_mask_history)
        prediction_history_all.append(prediction_history)

        n_evaluated_samples += features.shape[0]
        if only_n_samples is not None and n_evaluated_samples >= only_n_samples:
            break

    # Convert everything to tensors instead of lists of tensors
    # list[Tensor[budget,batch_size,n_features]] (n_batches)
    temp = [t.permute(1, 0, 2) for t in feature_mask_history_all]  # list[Tensor[batch_size,budget,n_features]] (n_batches)
    feature_mask_history_tensor: Tensor = torch.cat(temp)  # Tensor[n_samples,budget,n_features]

    # list[Tensor[budget,batch_size,n_classes]]] (n_batches)
    temp = [t.permute(1, 0, 2) for t in prediction_history_all]  # list[Tensor[batch_size,budget,n_classes]] (n_batches)
    prediction_history_tensor: Tensor = torch.cat(temp)  # Tensor[n_samples,budget,n_classes] (n_batches)

    # list[Tensor[batch_size,n_classes]] (n_batches)
    labels_tensor: Tensor = torch.cat(labels_all)  # Tensor[n_samples,C]

    labels_tensor = torch.argmax(labels_tensor, dim=1)

    accuracy_all, f1_all, bce_all = aggregate_metrics(
        prediction_history_tensor.cpu(), labels_tensor.cpu()
    )

    return {
        "accuracy_all": accuracy_all.detach().cpu(),
        "f1_all": f1_all.detach().cpu(),
        "bce_all": bce_all.detach().cpu(),
        "feature_mask_history_all": feature_mask_history_tensor,
        #     [t.detach().cpu() for t in sublist] for sublist in feature_mask_history_tensor
        # ],
    }

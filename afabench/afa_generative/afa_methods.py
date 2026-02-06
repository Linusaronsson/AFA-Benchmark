import math
from pathlib import Path
from typing import cast, override

import torch

from afabench.common.custom_types import (
    AFAMethod,
    AFAAction,
    AFAClassifier,
    FeatureMask,
    SelectionMask,
    Label,
    MaskedFeatures,
)
from afabench.common.bundle import load_bundle


class Ma2018AFAMethod(AFAMethod):
    def __init__(
        self,
        sampler,
        predictor,
        num_classes,
        device=torch.device("cpu"),
        lambda_threshold: float | None = None,
        selection_costs: torch.Tensor | None = None,
        n_contexts: int | None = None,
        num_mc_samples: int = 128,
        classifier_bundle_path: Path | None = None,
    ):
        super().__init__()
        self.sampler = sampler
        self.predictor = predictor
        self.num_classes = num_classes
        self._device: torch.device = device
        if lambda_threshold is None:
            self.lambda_threshold: float = -math.inf
        else:
            self.lambda_threshold = lambda_threshold
        self._selection_costs = selection_costs
        self.n_contexts = n_contexts
        self.num_mc_samples = num_mc_samples
        self.classifier = None
        if classifier_bundle_path is not None:
            classifier, _ = load_bundle(
                classifier_bundle_path, device=self._device
            )
            classifier = cast(AFAClassifier, cast(object, classifier))
            self.classifier = classifier

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.size(1) > 1:
            return logits.softmax(dim=1)
        probs = logits.sigmoid().view(-1, 1)
        return torch.cat([1 - probs, probs], dim=1)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        B, F = masked_features.shape
        zeros_label = torch.zeros(B, self.num_classes, device=self._device)
        zeros_mask = torch.zeros(
            B, self.num_classes, device=self._device, dtype=feature_mask.dtype
        )
        augmented_masked_feature = torch.cat(
            [masked_features, zeros_label], dim=-1
        ).to(self._device)
        augmented_feature_mask = torch.cat(
            [feature_mask, zeros_mask], dim=-1
        ).to(self._device)
        x_rep = augmented_masked_feature.repeat_interleave(self.num_mc_samples, dim=0)
        m_rep = augmented_feature_mask.repeat_interleave(self.num_mc_samples, dim=0)

        with torch.no_grad():
            _, _, _, z, _ = self.sampler(x_rep, m_rep)
            logits = self.predictor(z)
            probs = logits.softmax(dim=-1)
            probs = probs.view(B, self.num_mc_samples, -1).transpose(0, 1).contiguous()
            probs_mean = probs.mean(dim=0)
        return probs_mean

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        device = self._device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        B, F = masked_features.shape
        S = self.num_mc_samples
        zeros_label = torch.zeros(B, self.num_classes, device=self._device)
        zeros_mask = torch.zeros(
            B, self.num_classes, device=self._device, dtype=feature_mask.dtype
        )
        augmented_masked_feature = torch.cat(
            [masked_features, zeros_label], dim=-1
        ).to(self._device)
        augmented_feature_mask = torch.cat(
            [feature_mask, zeros_mask], dim=-1
        ).to(self._device)

        with torch.no_grad():
            x_rep = augmented_masked_feature.repeat_interleave(S, dim=0)
            m_rep = augmented_feature_mask.repeat_interleave(S, dim=0)
            _, _, _, z_base, x_full = self.sampler.forward(
                x_rep, m_rep
            )
            if self.classifier is None:
                logits_base = self.predictor(z_base)
                probs_base = self._logits_to_probs(logits_base).view(S, B, -1)
                base_probs = probs_base.mean(dim=0)
            else:
                # TODO: make sure classifier always returns probabilities
                base_probs = self.classifier(
                    masked_features=masked_features,
                    feature_mask=feature_mask,
                    feature_shape=feature_shape,
                )
        x_full = x_full.view(S, B, -1)[:, :, :F]
        missing = ~feature_mask.bool()
        x_filled = masked_features.unsqueeze(0).expand(S, B, F).clone()
        x_filled[:, missing] = x_full[:, missing]
        # (S, B, F+num_classes)
        x_full = torch.cat(
            [x_filled, zeros_label.unsqueeze(0).expand(S, B, -1)],
            dim=-1
        )
        if selection_mask is not None:
            n_sel = selection_mask.shape[1]
        else:
            # direct unmasker fallback
            n_sel = F
        if n_sel == F:
            add_masks = torch.eye(F, device=device, dtype=torch.bool)
        else:
            if self.n_contexts is None:
                raise ValueError("n_contexts must be set when using context selection.")
            expected = 1 + (F - self.n_contexts)
            if n_sel != expected:
                raise ValueError(f"Got n_sel={n_sel}, expected {expected} for n_contexts={self.n_contexts}.")
            add_masks = torch.zeros(n_sel, F, device=device, dtype=torch.bool)
            add_masks[0, : self.n_contexts] = True
            rem = F - self.n_contexts
            add_masks[1:, self.n_contexts:] = torch.eye(rem, device=device, dtype=torch.bool)

        # feature_indices = torch.eye(
        #     F, device=device, dtype=torch.bool
        # )
        mask_features_all = feature_mask.bool().unsqueeze(
            1
        ) | add_masks.unsqueeze(0)
        mask_features_flat = mask_features_all.reshape(B * n_sel, F).to(feature_mask.dtype)
        mask_label_all = zeros_mask.unsqueeze(1).expand(B, n_sel, -1)
        mask_label_flat = mask_label_all.reshape(B * n_sel, self.num_classes)
        # (B*n_sel, (F+num_classes))
        mask_tests = torch.cat([mask_features_flat, mask_label_flat], dim=1)
        # (S*B*n_sel, (F+num_classes))
        mask_tests_rep = mask_tests.unsqueeze(0).expand(S, -1, -1).reshape(S * B * n_sel, F + self.num_classes)
        x_rep = x_full.unsqueeze(2).expand(S, B, n_sel, F + self.num_classes)
        x_rep = x_rep.reshape(S * B * n_sel, F + self.num_classes)
        x_masks = x_rep * mask_tests_rep
        with torch.no_grad():
            if self.classifier is None:
                _, _, _, z_all, _ = self.sampler(x_masks, mask_tests_rep)
                logits_all = self.predictor(z_all)
                preds_all = self._logits_to_probs(logits_all).view(S, B * n_sel, -1)
            else:
                x_masks_raw = x_masks[:, :F]
                mask_tests_raw = mask_tests_rep[:, :F]
                preds_flat = self.classifier(
                    masked_features=x_masks_raw,
                    feature_mask=mask_tests_raw,
                    feature_shape=feature_shape,
                )
                preds_all = preds_flat.view(S, B * n_sel, -1)

        # S: num_mc_samples
        S, Bn, C = preds_all.shape
        # 1/n Σ p(y|x_s, x_i^j)
        base_probs_flat = (
            base_probs.unsqueeze(1)
            .expand(B, n_sel, C)
            .reshape(1, B * n_sel, C)
            .expand(S, B * n_sel, C)
        )
        # mean_preds = preds_all.mean(dim=0)
        # KL(p_s || mean), (S, B*n_sel)
        kl_all = (
            preds_all
            * ((preds_all + 1e-6).log() - (base_probs_flat + 1e-6).log())
        ).sum(dim=-1)
        kl_mean_flat = kl_all.mean(dim=0)

        scores = kl_mean_flat.view(B, n_sel)
        # avoid choosing the already masked features
        if selection_mask is not None:
            assert scores.shape == selection_mask.shape
            scores = scores.masked_fill(selection_mask.bool(), float("-inf"))
        else:
            assert scores.shape == feature_mask.shape
            scores = scores.masked_fill(feature_mask.bool(), float("-inf"))
        if self._selection_costs is not None:
            costs = self._selection_costs.to(self._device)
            costs = torch.clamp(costs, min=1e-12)
            scores = scores / costs.unsqueeze(0)
        best_scores, best_idx = scores.max(dim=1)
        lam = self.lambda_threshold
        stop_mask = best_scores < lam
        stop_mask = stop_mask | (best_scores < -1e5)
        selections = (best_idx + 1).to(torch.long).unsqueeze(-1)
        selections = selections.masked_fill(stop_mask.unsqueeze(-1), 0)
        return selections

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(
            str(path / "model.pt"), map_location=device, weights_only=False
        )
        sampler = checkpoint["sampler"]
        predictor = checkpoint["predictor"]
        num_classes = checkpoint["num_classes"]
        n_contexts = checkpoint.get("n_contexts", None)
        lambda_threshold = checkpoint.get("lambda_threshold", None)
        selection_costs = checkpoint.get("selection_costs", None)
        feature_costs = checkpoint.get("feature_costs", None)
        if selection_costs is not None:
            selection_costs = selection_costs.to(device)
        elif feature_costs is not None:
            selection_costs = feature_costs.to(device)

        method = cls(
            sampler=sampler,
            predictor=predictor,
            num_classes=num_classes,
            device=device,
            lambda_threshold=lambda_threshold,
            selection_costs=selection_costs,
            n_contexts=n_contexts,
        )
        classifier = checkpoint.get("classifier", None)
        if classifier is not None:
            classifier = cast(AFAClassifier, cast(object, classifier))
            classifier = classifier.to(device)
            method.classifier = classifier
        return method.to(device)

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sampler": self.sampler,
                "predictor": self.predictor,
                "classifier": self.classifier,
                "num_classes": self.num_classes,
                "lambda_threshold": float(self.lambda_threshold),
                "selection_costs": (
                    self._selection_costs.detach().cpu()
                    if self._selection_costs is not None
                    else None
                ),
                "n_contexts": self.n_contexts,
            },
            str(path / "model.pt"),
        )

    def to(self, device):
        self.sampler = self.sampler.to(device)
        self.predictor = self.predictor.to(device)
        if self.classifier is not None:
            self.classifier = self.classifier.to(device)
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    def cost_param(self) -> float | None:
        return float(self.lambda_threshold)

    def set_cost_param(self, cost_param: float) -> None:
        self.lambda_threshold = cost_param

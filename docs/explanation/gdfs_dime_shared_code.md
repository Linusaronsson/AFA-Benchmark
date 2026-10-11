# What GDFS and DIME share, and where the port departs from upstream

GDFS (Covert et al., ICML 2023) and DIME (Gadgil, Covert and Lee, 2023)
come from the same group, and their reference implementations live in
one upstream code base where DIME's trainer was built on GDFS's. The
benchmark's port inherited that shape as two near-identical copies.
Issue #105 folded the copies into shared code and, as that issue
required, checked the shared parts against the upstream implementations
([`iancovert/dynamic-selection`](https://github.com/iancovert/dynamic-selection)
`dynamic_selection/greedy.py` and
[`suinleelab/DIME`](https://github.com/suinleelab/DIME)
`dime/cmi_estimator.py`). This page records what is shared, why, and
every behavioural difference from upstream that the check found.

## What is shared

- **The checkpoint**, `NetworkPairSpec` in
  `src/afabench/components/methods/discriminative/common/checkpoint.py`.
  Both methods save a policy network and a predictor to `model.pt` with an
  `architecture` dict; the dict's keys are unchanged, and the policy
  network's hidden layers stay keyed by each method's own name for it
  (`selector_hidden_layers`, `value_network_hidden_layers`), so published
  bundles load as before.
- **The method class**, `NetworkPairAFAMethod` in
  `discriminative/common/afa_methods.py`: `save`, `load`, `to`, the
  soft-budget parameter (`lambda_threshold`), `predict`, the exclusion of
  performed selections and the scaling of scores by selection cost.
  `GDFSAFAMethod` and `DIMEAFAMethod` keep only their own `act`: GDFS ranks
  the selector's logits and stops on the predictor's entropy; DIME ranks
  the value network's CMI estimates and stops when none is worth its cost.
- **The training loop**, `NetworkPairTrainer` in
  `discriminative/common/trainer.py`: the validation metric protocol
  (`val_loss_fn` with `val_loss_mode`, defaulting to the training loss
  minimised), Adam with `ReduceLROnPlateau`, the best-epoch copies and the
  early-stopping counter, per-epoch metric logging, and the batch
  episodes that the Initializer starts and the Unmasker advances. Each
  paper's step is its `TrainingObjective`: `GDFSTrainingObjective` (a
  Concrete relaxation of the selection, one optimizer per temperature,
  the zero-temperature best kept at the end) and `DIMETrainingObjective`
  (CMI regression with epsilon-greedy exploration, decaying epsilon and
  resetting the learning rate on a plateau).
- **The modality setup**, `prepare_modality` in
  `discriminative/common/modality.py` (issue #106), so each method has one
  `train` for tabular and image datasets. It reads the pretrained model's
  architecture and answers a `TabularSetup` (MLP policy network, flat
  features with the feature mask appended) or an `ImageSetup` (ConvNet
  policy network on a ResNet backbone, patches of the pretrained model's
  geometry): the loaders, the mask layer, the policy network and its
  output check, and the checkpoint's `NetworkPairSpec`. The per-modality
  differences the trainers always had stay in the setup: only tabular
  training weights classes, charges the dataset's feature costs and
  falls back to the upstream `min_lr` when the config sets none; image
  training charges unit costs and requires `min_lr`. DIME passes its
  predictor to `policy_network(share_with=...)`, which ties the first
  two linear layers (tabular) or reuses the backbone (image).

Only what upstream GDFS and DIME already had in common is shared; each
difference between the two papers stays in the method's own package.

## Equivalence with the previous port

With the same seeds the shared loop trains the same parameters as the
replaced `GreedyDynamicSelection.fit` and `CMIEstimator.fit`, bit for
bit, for both methods with the direct and the Context Unmasker and with
cold and warm starts, after one change in RNG order: the old `fit`
created two `DataLoader` iterators over the validation set before
training, and every iterator creation draws a base seed from torch's
global generator; the trainer creates one. ADR 0001 allows that
(resolved hyperparameters identical, bit-identical bundles not
required). Resolved training hyperparameters are unchanged for every
GDFS and DIME config.

## Departures from upstream in the shared code

These predate #105 and are kept on purpose, because #105 is a
behaviour-preserving refactor; each is the maintainers' call.

- **DIME steps its tied parameters twice per optimizer update**
  (#116). The port ties the value network's first linear layers to the
  predictor's (tabular) and builds the value network on the predictor's
  backbone (image), then lists both networks' parameters in Adam, so a
  tied parameter appears twice. Torch 2.7 updates a duplicated parameter
  twice per `step()` (verified: exactly twice the single-listing delta)
  and warns that this will become an error. Upstream GDFS builds
  `Adam(set(...))` and upstream DIME `Adam(self.parameters())`, both of
  which deduplicate. GDFS shares no parameters, so only DIME is affected.
  `test_dime_lists_its_tied_parameters_twice_in_the_optimizer` pins the
  current behaviour.
- **DIME's scheduler and best-epoch check use the validation metric.**
  Upstream DIME steps `ReduceLROnPlateau` on the predictor's validation
  loss in `min` mode whatever `val_loss_fn` is; the port's DIME steps on
  the mean of `val_loss_fn` over the episode in `val_loss_mode`, as GDFS
  does. The pipeline sets no validation metric for DIME, so the metric
  is the training loss and the two agree.
- **Selection costs.** Upstream GDFS has no feature costs; the port
  subtracts the log selection cost from the selector's logits in
  training, so that the Concrete sample ranks `exp(logit) / cost`, and
  divides the logits by the selection costs at evaluation. Upstream DIME
  divides CMI estimates by feature costs; the port divides by the
  selection costs the Unmasker derives from the feature costs, clamped
  to at least `1e-12`. The costs are computed once in the shared loop
  and used per method.
- **Episodes start from the Initializer and advance through the
  Unmasker**, so both methods train on feature groups (image patches,
  the CUBE-NM context) and on warm starts; upstream starts every episode
  from an empty mask and selects single features.
- **DIME's exploration draws from torch's generator** (`torch.randint`)
  where upstream draws from NumPy's, and the port runs one backward pass
  per batch instead of one per selection, which
  `test_dime_single_backward_matches_per_acquisition_backward` shows
  gives the same gradients.

## A pre-existing bug kept for compatibility

`load` rebuilds an image method's predictor and policy network on one
ResNet backbone and loads the policy network's state dict first, so the
predictor's backbone weights overwrite the policy network's. GDFS trains
the selector on its own backbone, so a loaded GDFS image bundle runs the
selector on the predictor's backbone (#117). DIME trains the value
network on the predictor's backbone, so for DIME the rebuilt pair
matches training. `NetworkPairSpec.build_networks` keeps the shared
backbone because published GDFS image bundles must load and evaluate
unchanged.

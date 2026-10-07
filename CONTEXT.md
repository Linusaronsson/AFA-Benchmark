# AFABench

A benchmark framework for Active Feature Acquisition (AFA): comparing methods
that decide, per instance and one step at a time, which costly features to
acquire before predicting a class label. The vocabulary below follows the
accompanying paper (Schütz et al., KDD '26, arXiv:2508.14734) where the paper
and code agree, and the code where they differ. Paper-only synonyms are listed
under _Avoid_.

## Language

### Problem setting

**Active Feature Acquisition (AFA)**:
Sequentially choosing which features of a single instance to acquire, trading
predictive performance against acquisition cost. Decisions are instance-wise
and may condition on previously observed values.
_Avoid_: Dynamic feature selection, sequential feature selection, active
learning (that acquires labels, not features)

**Static feature selection**:
Choosing one global subset of features used for every instance. Included in the
benchmark only as a baseline paradigm.
_Avoid_: Global feature selection, feature importance method

**Instance**:
One data point: a full feature vector together with its label. In an image
dataset an instance is a whole image.
_Avoid_: Sample, example, row

**Feature**:
One scalar entry of an instance. For images, one pixel channel value.
_Avoid_: Attribute, variable, column

**Feature group**:
A set of features that is always revealed together by a single selection, for
example an image patch or the one-hot context columns of CUBE-NM. With the
direct Unmasker every feature group is a single feature.
_Avoid_: Patch (use only for images), block (reserved for CUBE-NM)

**Feature cost**:
The fixed, dataset-defined non-negative cost of acquiring one feature. Uniform
unit cost is the default.
_Avoid_: Price, acquisition fee

**Selection cost**:
The cost incurred by one selection, derived by the Unmasker from the feature
costs of the group it reveals.
_Avoid_: Action cost

**Accumulated cost**:
The sum of selection costs paid so far in an episode.
_Avoid_: Cumulative cost, spent budget

### Episode components

**Episode**:
The evaluation of one instance: starting from the initial feature mask, the
policy acts and the Unmasker reveals features step by step until a stop. Each
round is a **time step**.
_Avoid_: Trajectory, rollout, run

**Feature mask**:
A boolean indicator per feature of whether it has been observed.
_Avoid_: Observation mask, visibility mask

**Masked features**:
The instance's features with every unobserved feature replaced by zero.
_Avoid_: Partial observation, observed subset

**Initializer**:
The component that chooses the initial feature mask of an episode. A **cold
start** reveals nothing; a **warm start** reveals a fixed or random subset.
_Avoid_: Warm-up, initial policy

**Unmasker**:
The component that maps a selection to the feature group it reveals and
reports the selection costs. Changing the acquisition scheme of a dataset
means writing a new Unmasker, not a new method.
_Avoid_: Revealer, acquisition function, mask updater

**Direct Unmasker**:
The Unmasker where selection i reveals exactly feature i.
_Avoid_: Identity unmasker, default unmasker

**Image patch Unmasker**:
The Unmasker where each selection reveals one square patch of an image.
_Avoid_: Patch unmasker, grid unmasker

**Context Unmasker**:
The CUBE-NM Unmasker, which has one selection that reveals all one-hot context
features at once.
_Avoid_: CUBE-NM unmasker (as a concept; it is fine as a config name)

**Policy**:
The part of an AFA method that outputs an action given the current masked
features and feature mask.
_Avoid_: Agent, selector, acquirer

**Classifier**:
A model that predicts the label from masked features and a feature mask. The
benchmark is classification-only.
_Avoid_: Predictor (paper term), model (ambiguous)

**External classifier**:
A classifier pretrained independently of any method and shared by all methods
on a dataset during evaluation. Headline results use it.
_Avoid_: Shared classifier, common classifier

**Built-in classifier**:
A classifier that an AFA method trains and ships itself, often jointly with its
policy. Reported separately from external-classifier results.
_Avoid_: Internal predictor, internal classifier (paper terms), joint
classifier

**AFA method**:
A policy together with its optional built-in classifier, packaged so it can be
trained, saved, loaded, and evaluated.
_Avoid_: Algorithm, model, approach

### Actions and selections

**Action**:
The policy's output at a time step: either stop, or a request to reveal one
feature group. Action 0 is stop; action i > 0 is selection i - 1.
_Avoid_: Query, move, decision

**Stop action**:
The action that ends the episode and triggers the final prediction.
_Avoid_: Terminate, halt, done

**Selection**:
A zero-indexed choice of feature group handed to the Unmasker. Selections never
include stop, so there is one fewer selection than action.
_Avoid_: Feature index (not the same once groups exist), choice

**Selection history**:
The ordered sequence of selections performed before a time step in an episode,
including repetitions but excluding stop and features revealed by the
Initializer.
_Avoid_: Action history (includes stop), acquired-feature set (loses order)

**Selection mask**:
A boolean indicator per selection of whether it has already been performed.
_Avoid_: Action mask, availability mask

**Forced stop**:
An episode ending because the next action would have exceeded the hard budget,
rather than because the policy chose to stop.
_Avoid_: Budget cutoff, truncation

**Forced acquisition**:
Overriding stop actions with the first available selection so that a policy
keeps acquiring up to the hard budget. Used in hard-budget evaluation.
_Avoid_: Ignore stop, no-stop mode

**Cheating method**:
A method or component that is handed the true label during an episode, for
benchmarking upper bounds only.
_Avoid_: Oracle (reserved for the oracle-based paradigm)

### Budgets

**Hard budget**:
A per-instance cap on accumulated selection cost. With unit costs it equals
the number of allowed selections, not the number of features.
_Avoid_: Feature budget, max features, k

**Soft-budget parameter**:
The scalar that weights acquisition cost against prediction loss and so lets a
policy spend more on hard instances. May differ between training and
evaluation.
_Avoid_: Alpha, lambda, cost parameter, trade-off coefficient

**Hard-budget setting**:
The evaluation regime where every method acquires up to the same hard budget
and no soft-budget parameter is given.
_Avoid_: Fixed-budget setting, k-feature setting

**Soft-budget setting**:
The evaluation regime where no hard budget is enforced and each method stops
according to its soft-budget parameter.
_Avoid_: Early-stopping setting, cost-sensitive setting

### Method taxonomy

**Myopic method**:
A method that picks the next feature by maximizing one-step expected
information gain about the label per unit cost, with no lookahead.
_Avoid_: Greedy (README legacy), one-step method

**Non-myopic method**:
A method that optimizes long-term utility and may pay for uninformative
features now to acquire better ones later.
_Avoid_: Non-greedy, planning method, lookahead method

**Conditional mutual information (CMI)**:
The quantity myopic methods estimate: information a candidate feature carries
about the label given the observed features. Methods are split by whether they
estimate it **generatively** or **discriminatively**.
_Avoid_: Information gain (as a distinct concept), utility

**Model-free RL method**:
A non-myopic method that learns a policy directly from episodes of the AFA
Markov decision process.
_Avoid_: Policy-gradient method, Q-learning method (too specific)

**Model-based RL method**:
A non-myopic method that learns an explicit model of unobserved features and
plans with it.
_Avoid_: Planning-based RL

**Oracle-based method**:
A non-myopic method that scores candidate acquisitions with a lookahead oracle
instead of reinforcement learning.
_Avoid_: Non-RL method, search-based method

**Dummy method**:
A method with trivial behavior (random or sequential selection) used as a
sanity check and lower bound.
_Avoid_: Baseline (static methods are also baselines), toy method

### Datasets

**Dataset key**:
The canonical snake_case string identifying a dataset throughout configs,
file names, and feature-cost files.
_Avoid_: Dataset name, dataset id

**Split**:
One of the train, validation, or test partitions of a dataset.
_Avoid_: Fold, subset

**Dataset instance**:
One seeded generation and split of a dataset. Benchmark results average over
several dataset instances.
_Avoid_: Seed (the seed is the input, the instance is the output), run

**CUBE**:
The synthetic tabular AFA dataset where each class makes three class-specific
features informative and the rest noise.
_Avoid_: Cube dataset (capitalize)

**CUBE-NM**:
The synthetic dataset introduced by this project, which prepends a **context
feature** that selects which **block** of CUBE-style features is informative.
Built to expose the gap between myopic and non-myopic methods.
_Avoid_: AFAContext (legacy class name), cube context

**CUBE-NUC**:
The CUBE variant with non-uniform feature costs: the informative features are
duplicated in a second block that costs twice as much.
_Avoid_: Cube non-uniform, costly cube

**Noiseless variant**:
A synthetic dataset generated with zero noise, so perfect accuracy is
attainable and the optimal policy is known.
_Avoid_: Clean variant, deterministic variant

### Pipeline

**Benchmark release**:
A curated, versioned collection of benchmark results and reusable outputs
produced with an identified code revision and pipeline configuration.
_Avoid_: Latest results, pipeline run (a run need not be published)

**Output snapshot**:
A verbatim copy of everything the pipeline has written under its output
root, taken so it can later be put back exactly where the workflow expects
it. It is not curated; a benchmark release is built from one. Its only
provenance is an optional release manifest beside it.
_Avoid_: Backup, archive, export, package

**Release manifest**:
The JSON file beside an output snapshot's output tree that identifies its
benchmark release, declares full, partial or test-only scope, and records
the producing commit, workflow configuration, resolved settings, per-table
identity and coverage (see `docs/release_manifest.md`). Smoke-test outputs
are always test-only.
_Avoid_: Metadata, release info, provenance record (that is per artifact)

**Payload category**:
One kind of reusable output a release manifest lists: raw or transformed
evaluation tables, or dataset, classifier, pretrained-model or AFA-method
bundles. Dataset bundles, external classifiers and pretrained models are
**shared prerequisites** of any method; AFA-method bundles are optional
baselines, never needed to plot against published results.
_Avoid_: Artifact type, output kind

**Test release**:
A test-only output snapshot published apart from benchmark releases, to
check the publish/download round trip. It is never a benchmark release and
is downloaded only by asking for a test release (see
`docs/release_publishing.md`).
_Avoid_: Smoke release, staging release

**Reference method**:
A method name whose plotting-ready evaluation tables are restored from a
benchmark release and compared with locally produced methods, without the
workflow ever training, evaluating or transforming it.
_Avoid_: Baseline method (any compared method can be a baseline),
downloaded method

**Pipeline stage**:
One of **pretraining**, **training**, and **evaluation**. The first two are
optional per method; evaluation is mandatory and shared by all methods.
_Avoid_: Phase, step (reserved for time steps)

**Execution activity**:
A kind of pipeline job whose hardware the workflow resolves. Classifier
training, pretraining, training and evaluation each take a declared `cpu` or
`cuda` choice; dataset generation, transformation, aggregation and
visualization always run on CPU.
_Avoid_: Stage (for classifier training or processing jobs)

**Pretrained model**:
A named artifact produced in the pretraining stage and reusable across
methods, for example a partial VAE shared by EDDI and ODIN.
_Avoid_: Base model, backbone, checkpoint

**Method name**:
The pipeline-level identifier of one configured variant of an AFA method, such
as a method paired with its external or built-in classifier.
_Avoid_: Method id, method key

**Method set**:
A named group of method names plotted together.
_Avoid_: Method group, plot group

**Bundle**:
The on-disk folder format in which datasets, classifiers, pretrained models,
and AFA methods are saved and loaded.
_Avoid_: Checkpoint, artifact (as the generic term), pickle

**Provenance record**:
The typed description of how one bundle or evaluation table was produced:
code commit, resolved configuration, seed actually used, input bundles and
their content hashes, environment, and dataset identity. It is embedded in
the artifact itself (see `docs/adr/0002-provenance-recorded-in-artifacts.md`).
_Avoid_: Metadata (the free-form manifest field), lineage, run info

**Smoke test**:
A run mode where every stage executes as fast as possible to verify the
pipeline works end to end.
_Avoid_: Dry run, quick mode

**Training contract**:
The fixed set of inputs the pipeline gives a pretraining or training script,
and the bundle it expects back at the save path. Pretraining receives a
subset.
_Avoid_: Script interface, pipeline arguments, CLI args

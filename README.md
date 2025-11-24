# AFA Benchmark
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License:](https://img.shields.io/badge/License-XYZ-yellow.svg)]()
[![Paper](https://img.shields.io/badge/KDD%202026-Paper-red.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2508.14734-b31b1b.svg)](https://arxiv.org/abs/2508.14734)

**A comprehensive benchmark for Active Feature Acquisition (AFA) methods**

Compare state-of-the-art algorithms for sequential feature selection in
scenarios where acquiring features is costly. Includes implementations of
multiple AFA methods, standardized datasets, and automated evaluation pipelines.

## Latest Updates
**2025-10-16**:
Trained and evaluated methods for the soft budget case. To reduce clutter, here is a plot with only a subset of the datasets, and only showing the accuracy/F1 score of the external classifier:
![](result/soft_budget_5_splits.png)
Note that the Diabetes dataset uses non-uniform acquisition costs, and that physionet uses f1 score instead of accuracy.

## Features
- Easily readable and reproducible configuration using
  [hydra](https://hydra.cc/) and [snakemake](https://snakemake.readthedocs.io/en/stable/).
- Modular design: rerun specific parts of the pipeline as needed.
- Extensible framework: add custom datasets and AFA methods.

## Limitations
- Supports only classification tasks; regression tasks are not yet implemented.

## What is Active Feature Acquisition?
**Active Feature Acquisition (AFA)** addresses scenarios where,

- **Features are expensive** to obtain (medical tests, surveys, sensors),
- **Real-time decisions** must be made with partial information,
- **Budget constraints** limit which features you can acquire.

**Example**: Medical diagnosis where each test costs money and time. AFA methods
intelligently decide which tests to order next based on previous results, aiming
for accurate diagnosis with minimal cost.

## Installation

### Prerequisites
- [uv](https://docs.astral.sh/uv/)
- [Weights & Biases](https://wandb.ai) account (for experiment tracking)
- [mprocs](https://github.com/pvolok/mprocs) (optional, for batch training)

### Setup
```bash
# Clone repository
git clone https://github.com/Linusaronsson/AFA-Benchmark.git
cd AFA-Benchmark

# Install dependencies with uv
uv sync

# Setup W&B
uv run wandb login
```

Additionally, if you have access to a cluster running
[slurm](https://slurm.schedmd.com/), you might be interested in adding a
configuration file to the `conf/global/hydra/launcher/` directory. The name of
this file can then be referenced in scripts in order to run experiments in
parallel.

## Simple Example
Train and evaluate a single AFA method on synthetic data:

```bash
# 1. Generate all 5 splits of the cube dataset
uv run scripts/dataset_generation/generate_dataset.py dataset_type=cube

# 2. Train the MLP classifier on split 1
uv run scripts/train_classifiers/train_masked_mlp_classifier.py dataset_artifact_name=cube/cube_split_1

# 3.0 Train AACO method
uv run scripts/train_methods/train_aaco.py dataset_artifact_name=cube/cube_split_1

# 3.1 Pretrain Covert2023 method
uv run scripts/pretrain/covert2023.py

# 3.2 Train Covert2023 method
uv run scripts/train/covert2023.py

# 3.3 Pretrain Kachuee2019 method
uv run scripts/pretrain/kachuee2019.py

# 3.1 Train Kachuee2019 method
uv run scripts/train/kachuee2019.py cost_param=0.001 hard_budget=null

# 4. Evaluate performance
uv run scripts/eval/eval_soft_afa_method.py trained_method_artifact_name=train_kachuee2019_cube_split_1_costparam_0.001_seed_42 trained_classifier_artifact_name=masked_mlp_classifier-cube_split_1 cost_param=0.001 batch_size=64
```

## Implemented Methods

|    Method     |                                                                            Paper                                                                             |             Strategy             |  Greedy?   |
| :-----------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------: | :--------: |
|  **EDDI-GG**  |                                                       [link](https://proceedings.mlr.press/v97/ma19c)                                                        |   Generative estimation of CMI   |   Greedy   |
|  **GDFS-DG**  |                                                     [link](https://proceedings.mlr.press/v202/covert23a)                                                     | Discriminative estimation of CMI |   Greedy   |
|  **DIME-DG**  |                                                           [link](https://arxiv.org/pdf/2306.03301)                                                           | Discriminative estimation of CMI |   Greedy   |
| **JAFA-MFRL** |                          [link](https://papers.nips.cc/paper_files/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html)                           |          Model-free RL           | Non-greedy |
|  **OL-MFRL**  |                                                           [link](https://arxiv.org/pdf/1901.00243)                                                           |          Model-free RL           | Non-greedy |
| **ODIN-MFRL** | [link](https://www.microsoft.com/en-us/research/publication/odin-optimal-discovery-of-high-value-information-using-model-based-deep-reinforcement-learning/) |          Model-free RL           | Non-greedy |
| **ODIN-MBRL** | [link](https://www.microsoft.com/en-us/research/publication/odin-optimal-discovery-of-high-value-information-using-model-based-deep-reinforcement-learning/) |          Model-based RL          | Non-greedy |
|   **AACO**    |                                                 [link](https://proceedings.mlr.press/v235/valancius24a.html)                                                 |           Oracle-based           | Non-greedy |
|   **PT-S**    |                                              [link](https://link.springer.com/article/10.1023/A:1010933404324)                                               |    Global feature importance     |    N/A     |
|   **CAE-S**   |                                                   [link](https://proceedings.mlr.press/v97/balin19a.html)                                                    |    Global feature importance     |    N/A     |

## Datasets

|     Dataset      |    Type    | Size (total, # data instances) | # Features | # Classes |
| :--------------: | :--------: | :----------------------------: | :--------: | :-------: |
|     **CUBE**     | Synthetic  |              1000              |     20     |     8     |
|  **AFAContext**  | Synthetic  |              1000              |     30     |     8     |
|    **MNIST**     | Real World |             60 000             |    784     |    10     |
| **FashionMNIST** | Real World |             60 000             |    784     |    10     |
|   **Diabetes**   | Real World |             92 063             |     45     |     3     |
|  **PhysioNet**   | Real World |             12 000             |     41     |     2     |
|  **MiniBooNE**   | Real World |            130 064             |     50     |     2     |

## Project structure
- `afabench`: Main package.
- `docs`: Documentation.
- `extra`: Saved methods, data, logs and so on, non-source code files.
    - `classifiers`: Saved common classifier models.
    - `conf`: This is where all the configuration files are. Each configuration file
      corresponds to a class in `config_classes.py`.
    - `data`: Where datasets are saved and loaded from.
    - `result`: Saved pretrained models, trained methods, evaluation results.
    - `workflow`: Snakemake workflows for running the full pipeline.
- `scripts/`:
  - `dataset_generation/generate_dataset.py`: A script that generates datasets
    individually. Generates a dataset artifact.
  - `evaluation/eval_afa_method.py`: Evaluates a single method with a hard
    budget on a single dataset split.
  - `evaluation/eval_soft_afa_method.py`: Evaluates a single method with a soft
    budget on a single dataset split.
  - `pretrain/`: Pretraining scripts for methods that require pretraining.
  - `train/`: Training scripts for methods.
- `result`: Where final plots are saved.
- `test`: Unit tests.

## Citation
If you use this benchmark in your research, please cite,

```bibtex
@misc{schütz2025afabenchgenericframeworkbenchmarking,
      title={AFABench: A Generic Framework for Benchmarking Active Feature Acquisition},
      author={Valter Schütz and Han Wu and Reza Rezvan and Linus Aronsson and Morteza Haghir Chehreghani},
      year={2025},
      eprint={2508.14734},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.14734},
}
```

## License

## Acknowledgments

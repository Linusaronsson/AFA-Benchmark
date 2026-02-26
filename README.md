# AFA Benchmark
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.14734-b31b1b.svg)](https://arxiv.org/abs/2508.14734)

**A comprehensive benchmark for Active Feature Acquisition (AFA) methods**

Compare state-of-the-art algorithms for sequential feature selection in
scenarios where acquiring features is costly. Includes implementations of
multiple AFA methods, standardized datasets, and automated evaluation pipelines.

## Installation

[uv](https://docs.astral.sh/uv/getting-started/installation/) is the only external dependency.

```bash
# Clone repository
git clone https://github.com/Linusaronsson/AFA-Benchmark.git
cd AFA-Benchmark

# Install dependencies with uv
uv sync
```

## Quickstart

To run the pipeline locally with 8 cores, execute the following command at the repo root. It should produce plots at `extra/output/plot_results/`.

```shell
WANDB_PROJECT=afabench uv run snakemake \
    -s extra/workflow/snakefiles/orchestration/pipeline.smk \
    all \
    --configfile \
      extra/workflow/conf/eval_hard_budgets.yaml \
      extra/workflow/conf/methods.yaml \
      extra/workflow/conf/method_sets.yaml \
      extra/workflow/conf/method_options.yaml \
      extra/workflow/conf/pretrain_mapping.yaml \
      extra/workflow/conf/soft_budget_params.yaml \
      extra/workflow/conf/unmaskers.yaml \
      extra/workflow/conf/classifier_names.yaml \
      extra/workflow/conf/datasets_main.yaml \
    --config \
      eval_dataset_split=test \
      "dataset_instance_indices=[0,1,2,3,4]" \
      smoke_test=false \
      use_wandb=true \
      device=cpu \
    --jobs 8
```

See the [pipeline explanation](docs/tutorials/pipeline_explanation.md) tutorial for details on how this pipeline works and how to customize it.

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
for accurate diagnosis with minimal cost. See the following survey for details: [AFA Survey](https://arxiv.org/abs/2502.11067) 

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
| Dataset | Type | Modality | Train Size | Val Size | Test Size | \# Features | \# Groups | \# Classes |
| :-----: | :--: | :------: | :--------: | :------: | :-------: | :---------: | :-------: | :--------: |
| CUBE | Synthetic | Tabular | 600 | 200 | 200 | 20 | | 20 | 8 |
| CUBE-NM | Synthetic | Tabular | 600 | 200 | 200 | 33 | 33 | 8 |
| MNIST | Real World | Image (tabularized) | 36,000 | 12,000 | 12,000 | 784 | 784 | 10 |
| FashionMNIST | Real World | Image (tabularized) | 36,000 | 12,000 | 12,000 | 784 | 784 | 10 |
| Diabetes | Real World | Tabular | 55,237 | 18,412 | 18,413 | 45 | 45 | 3 | |
| PhysioNet | Real World | Tabular | 7,200 | 2,400 | 2,400 | 41 | 41 | 2 |
| MiniBooNE | Real World | Tabular | 78,038 | 26,012 | 26,014 | 50 | 50 | 2 |
| ACTG175 | Real World | Tabular | 1,283 | 427 | 429 | 23 | 23 | | 2 |
| CKD | Real World | Tabular | 240 | 80 | 80 | 24 | 24 | 2 |
| BankMarketing | Real World | Tabular | 27,126 | 9,042 | 9,043 | 16 | 16 | 2 |
| Imagenette | Real World | Image | 5,681 | 3,788 | 3,925 | 150,528 | 196 | 10 |

## Project structure
- `afabench`: Main package.
- `docs`: Documentation.
- `extra`: Saved methods, data, logs and so on, non-source code files.
    - `conf`: This is where all the **script** configuration files are. Each configuration file
      corresponds to a class in `config_classes.py`.
    - `data`: Where miscellaneous files for datasets (e.g., CSVs, custom generated costs, etc.) are stored.
    - `workflow`: Snakemake workflows for running the full pipeline.
    - `output`: Folder where outputs from the snakemake pipeline are stored.
- `scripts/`: Folder of scripts, many of which are called from the snakemake pipeline.
- `test`: Tests.
  - `src`: Tests related to library code in `afabench`.
  - `scripts`: Tests related to specific scripts in `scripts`.

## Tutorials

Learn more in our tutorials:
  - [Pipeline explanation](docs/tutorials/pipeline_explanation.md)
  - [Adding a new dataset](docs/tutorials/add_dataset.md)
  - [Adding a new method](docs/tutorials/add_method.md)


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

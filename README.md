# AFABench
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.14734-b31b1b.svg)](https://arxiv.org/abs/2508.14734)
[![KDD 2026](https://img.shields.io/badge/KDD-2026-ff69b4.svg)](https://www.kdd.org/kdd2026/)

**A benchmark for Active Feature Acquisition (AFA) methods**

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

Local execution is not recommended for reproducing the full benchmark because
the pipeline generates a large number of jobs. If you still want to run it
locally with 8 CPU cores, execute this command from the repo root:

```shell
uv run snakemake \
    -s extra/workflow/snakefiles/orchestration/pipeline.smk \
    all \
    --configfile \
      extra/workflow/conf/eval_hard_budgets/all.yaml \
      extra/workflow/conf/methods/all.yaml \
      extra/workflow/conf/method_sets/all.yaml \
      extra/workflow/conf/method_options/all.yaml \
      extra/workflow/conf/pretrain_mappings/all.yaml \
      extra/workflow/conf/soft_budget_params/all.yaml \
      extra/workflow/conf/unmaskers/all.yaml \
      extra/workflow/conf/classifier_names/all.yaml \
      extra/workflow/conf/datasets/all.yaml \
    --config \
      device=cpu \
    --jobs 8
```

To reproduce the full benchmark results, use SLURM instead. See the
[reproducing full results](docs/tutorials/reproduce_full_results.md) tutorial
for the exact commands.

The missing-training-data study has a separate staged workflow; see
[missing-training-data experiments](docs/tutorials/missing_data_experiments.md).

## Features

- Accessible configuration using
  [hydra](https://hydra.cc/)
- Reproducible pipeline using [snakemake](https://snakemake.readthedocs.io/en/stable/).
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
for accurate diagnosis with minimal cost. See the following survey for details: [AFA Survey](https://arxiv.org/abs/2502.11067).

## Implemented Methods
|    Method     |                                                                            Paper                                                                             |             Strategy             |  Greedy?   |
| :-----------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------: | :--------: |
|  **EDDI**  |                                                       [link](https://proceedings.mlr.press/v97/ma19c)                                                        |   Generative estimation of CMI   |   Greedy   |
|  **GDFS**  |                                                     [link](https://proceedings.mlr.press/v202/covert23a)                                                     | Discriminative estimation of CMI |   Greedy   |
|  **DIME**  |                                                           [link](https://arxiv.org/pdf/2306.03301)                                                           | Discriminative estimation of CMI |   Greedy   |
| **JAFA** |                          [link](https://papers.nips.cc/paper_files/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html)                           |          Model-free RL           | Non-greedy |
|  **OL**  |                                                           [link](https://arxiv.org/pdf/1901.00243)                                                           |          Model-free RL           | Non-greedy |
| **ODIN** | [link](https://www.microsoft.com/en-us/research/publication/odin-optimal-discovery-of-high-value-information-using-model-based-deep-reinforcement-learning/) |          Model-free RL           | Non-greedy |
| **ODIN** | [link](https://www.microsoft.com/en-us/research/publication/odin-optimal-discovery-of-high-value-information-using-model-based-deep-reinforcement-learning/) |          Model-based RL          | Non-greedy |
|   **AACO**    |                                                 [link](https://proceedings.mlr.press/v235/valancius24a.html)                                                 |           Oracle-based           | Non-greedy |
|   **PT**    |                                              [link](https://link.springer.com/article/10.1023/A:1010933404324)                                               |    Global feature importance     |    N/A     |
|   **CAE**   |                                                   [link](https://proceedings.mlr.press/v97/balin19a.html)                                                    |    Global feature importance     |    N/A     |

## Datasets
| Dataset | Type | Modality | Train Size | Val Size | Test Size | \# Features | \# Groups | \# Classes |
| :-----: | :--: | :------: | :--------: | :------: | :-------: | :---------: | :-------: | :--------: |
| CUBE | Synthetic | Tabular | 600 | 200 | 200 | 20 | | 20 | 8 |
| CUBE-NM | Synthetic | Tabular | 600 | 200 | 200 | 55 | 51 | 8 |
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
  - [Reproducing full results](docs/tutorials/reproduce_full_results.md)
  - [Pipeline explanation](docs/tutorials/pipeline_explanation.md)
  - [Adding a new dataset](docs/tutorials/add_dataset.md)
  - [Adding a new method](docs/tutorials/add_method.md)

## Development

We encourage researchers to fork this repository and implement their own methods. Take a look at the [tutorials](#tutorials) to get started.

To follow repo conventions, run
```shell
uv run just qa
```
which will

- format code with `ruff format`
- do linting and formatting with `ruff check --fix`
- type checking with `basedpyright --warnings`
- run tests with `pytest`

## Citation
If you use this benchmark in your research, please cite,
```bibtex
@inproceedings{Schütz2026,
    author = {Sch{\"u}tz, Valter and Wu, Han and Rezvan, Reza and Aronsson,
              Linus and Haghir Chehreghani, Morteza},
    title = {AFABench: A Generic Framework for Benchmarking Active Feature
             Acquisition},
    year = {2026},
    booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
                 Discovery and Data Mining},
    url = {https://doi.org/10.1145/3770855.3817493},
}
```

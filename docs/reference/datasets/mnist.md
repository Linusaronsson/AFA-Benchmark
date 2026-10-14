# MNIST (`mnist`)

Handwritten digits. An instance is one 28 × 28 grayscale image, its
features the 784 pixel values, and its label the digit, 0–9.

## Source

The images are not committed to the repository: their provider states no
license (ADR 0014). Dataset generation downloads them through
torchvision's `MNIST` into `data/mnist/MNIST/raw/` when they are missing;
torchvision checks each file against the MD5 it pins, and AFABench pins
nothing of its own (#136).

| | |
| --- | --- |
| Original provider | Y. LeCun, C. Cortes and C. J. C. Burges, <http://yann.lecun.com/exdb/mnist/>, which in October 2026 serves an empty directory |
| Download | torchvision's mirror <https://ossci-datasets.s3.amazonaws.com/mnist/>, falling back to the original provider |
| Files | `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`, the 60,000 training images, and the test files, which AFABench does not use |
| SHA-256 | `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609` (`train-images-idx3-ubyte.gz`) and `3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c` (`train-labels-idx1-ubyte.gz`); torchvision checks their MD5s |
| License | None stated by the provider, on its archived page or elsewhere |

## References

- Y. LeCun, L. Bottou, Y. Bengio and P. Haffner. Gradient-based learning
  applied to document recognition. *Proceedings of the IEEE* 86,
  2278–2324 (1998).

## Instances and label

60,000 instances: MNIST's training set, in its order. AFABench does not use
MNIST's test set; splits are drawn from the training set.

| Index | Digit | Instances |
| --- | --- | --- |
| 0 | 0 | 5,923 |
| 1 | 1 | 6,742 |
| 2 | 2 | 5,958 |
| 3 | 3 | 6,131 |
| 4 | 4 | 5,842 |
| 5 | 5 | 5,421 |
| 6 | 6 | 5,918 |
| 7 | 7 | 6,265 |
| 8 | 8 | 5,851 |
| 9 | 9 | 5,949 |

## Features

The image's 784 pixels in row-major order, from the top-left pixel, each
the gray level scaled from 0–255 to 0–1 (0 is the background). A selection
reveals one pixel (`direct` unmasker).

## Preprocessing in AFABench

- No normalization beyond the scaling to 0–1.
- All feature costs are 1.
- Splits are the pipeline's: 60% train, 20% validation, 20% test of the
  60,000 images, for each dataset realization.
- The hard budgets are 10, 20 and 30 (`all` profile).

# Fashion-MNIST (`fashion_mnist`)

Zalando article images. An instance is one 28 × 28 grayscale image, its
features the 784 pixel values, and its label the article's class.

## Source

The provider's four files are committed to the repository unchanged (ADR
0014), where torchvision's `FashionMNIST` finds them: it checks each
against the MD5 it pins and extracts it, so generation needs no network.
Downloading them again is what torchvision does when they are missing.

| | |
| --- | --- |
| Provider | [Zalando Research, `zalandoresearch/fashion-mnist`](https://github.com/zalandoresearch/fashion-mnist) |
| Download | <http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/>, Zalando's bucket, through torchvision |
| Files | `data/fashion_mnist/FashionMNIST/raw/`: `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`, the 60,000 training images, and the two test files, which AFABench does not use but torchvision requires |
| SHA-256 | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` (`train-images-idx3-ubyte.gz`), `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` (`train-labels-idx1-ubyte.gz`), `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` (`t10k-images-idx3-ubyte.gz`), `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` (`t10k-labels-idx1-ubyte.gz`) |
| License | MIT License, Copyright 2017 Zalando SE |

The MIT License requires keeping its copyright and permission notice with
copies of the files; `data/fashion_mnist/LICENSE.md` holds it.

## References

- H. Xiao, K. Rasul and R. Vollgraf. Fashion-MNIST: a novel image dataset
  for benchmarking machine learning algorithms. arXiv:1708.07747 (2017).

## Instances and label

60,000 instances: Fashion-MNIST's training set, in its order, 6,000 per
class. AFABench does not use its test set; splits are drawn from the
training set.

| Index | Class |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

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

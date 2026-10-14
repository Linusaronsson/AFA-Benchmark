# Imagenette (`imagenette`)

Photographs of 10 easily classified ImageNet classes. An instance is one
image, its features the 3 × 224 × 224 pixel channel values, and its label
the image's class.

## Source

The images are not committed to the repository: ImageNet's terms grant
no right to redistribute them (ADR 0014). Dataset generation downloads
and extracts the archive into `data/imagenette/` when it is missing, and
does not check it. Dataset bundles hold only generation indices and
configuration, so every user reads the images from their own download.

| | |
| --- | --- |
| Provider | [fast.ai, `fastai/imagenette`](https://github.com/fastai/imagenette), made by Jeremy Howard: Imagenette 2, released 2019-12-06, in its 320 px version, each image's shortest side resized to 320 px |
| Download | <https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz> |
| Files | `data/imagenette/imagenette2-320/train/` (9,469 images) and `val/` (3,925 images), one folder per class |
| SHA-256 | `569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c` for the archive downloaded on 2026-10-14; generation does not check it |
| Original provider | [ImageNet](https://www.image-net.org/), whose images Imagenette selects and resizes |
| License | The `fastai/imagenette` repository is under the Apache License 2.0 and states no license for the images. ImageNet's [terms of access](https://www.image-net.org/download.php) allow use "only for non-commercial research and educational purposes" and make the user responsible for any "copies of copyrighted images" made from it |

## References

- J. Howard. Imagenette. <https://github.com/fastai/imagenette> (2019).
- J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei. ImageNet:
  a large-scale hierarchical image database. *CVPR* (2009).

## Instances and label

13,394 instances: the images of `train/`, then those of `val/`, each in
the order torchvision's `ImageFolder` lists them. The test split is always
`val/`; each dataset realization shuffles `train/` into 60% train (5,681
images) and the rest validation (3,788 images).

The classes are the folders, in sorted order:

| Index | Folder | Class | `train/` | `val/` |
| --- | --- | --- | --- | --- |
| 0 | `n01440764` | Tench | 963 | 387 |
| 1 | `n02102040` | English springer | 955 | 395 |
| 2 | `n02979186` | Cassette player | 993 | 357 |
| 3 | `n03000684` | Chain saw | 858 | 386 |
| 4 | `n03028079` | Church | 941 | 409 |
| 5 | `n03394916` | French horn | 956 | 394 |
| 6 | `n03417042` | Garbage truck | 961 | 389 |
| 7 | `n03425413` | Gas pump | 931 | 419 |
| 8 | `n03445777` | Golf ball | 951 | 399 |
| 9 | `n03888257` | Parachute | 960 | 390 |

## Features

The RGB image as a 3 × 224 × 224 tensor, each channel normalized with
ImageNet's mean (0.485, 0.456, 0.406) and standard deviation (0.229,
0.224, 0.225). The image is transformed each time it is read:

- Train split: a random crop of random size and aspect ratio, resized to
  224 × 224, and a random horizontal flip, so the features of an instance
  differ between reads.
- Validation and test splits: the shorter side resized to 256 px, then the
  central 224 × 224 crop.

A selection reveals a 16 × 16 pixel patch in all three channels: 196
feature groups on a 14 × 14 grid (`224x224_to_14x14` unmasker).

## Preprocessing in AFABench

- All feature costs are 1. A selection costs the mean of its patch's
  feature costs, so 1.
- The hard budgets are 5, 10 and 15 (`all` profile).

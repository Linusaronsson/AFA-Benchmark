Each instance of a dataset is represented by a folder with contents
- `metadata.json`: a json object with keys
  - `dataset_type`: see `docs/dataset_types.md`.
  - `split_idx`: distinguishes different dataset instances of the same class
- `train.pt`: a pytorch tensor file containing the training set
- `val.pt`: a pytorch tensor file containing the validation set
- `test.pt`: a pytorch tensor file containing the testing set

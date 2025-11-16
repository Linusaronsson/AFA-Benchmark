## Filename cleanup
```
scripts/pretrain_models/pretrain_shim2018.py → scripts/pretrain_models/shim2018.py
scripts/train_methods/train_aaco.py → scripts/train_methods/aaco.py
scripts/train_classifiers/train_masked_mlp_classifier.py → scripts/train_classifiers/masked_mlp.py
```

Then `SAVE_PATH` auto-generates: `result/shim2018/`, `result/aaco/`, etc.

## Final structure:
```
extra/
  data/{dataset}/
  classifiers/{classifier_type}/{dataset_variant}/
  {method}/
    {dataset}_{params}/
      method.pt
      metadata.json
      eval_*.csv
```

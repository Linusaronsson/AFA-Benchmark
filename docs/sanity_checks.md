## Pretraining models

We pretrained the models on synthetic datasets with no noise, to check whether they would get 100% accuracy as expected.

### shim2018

4011a6abf372

**Yes:**
- Noiseless Cube (100%)
- Noiseless AFAContext (100%)
- Noiseless synthetic MNIST (100%)

---

### zannone2019

4011a6abf372

**Yes:**
- Noiseless Cube (100%)
- Noiseless AFAContext (100%)
- Noiseless synthetic MNIST (100%)

### kachuee2019

4011a6abf372

**Yes:**
- Noiseless synthetic MNIST (99%)

**No:**
- Noiseless AFAContext (80-90%)
- Noiseless Cube (95%)

Since kachuee2019 is not able to distinguish between missing features and features with values 0, these results are not surprising.

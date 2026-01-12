## Pretraining models

We pretrained the models on synthetic datasets with no noise, to check whether they would get 100% accuracy as expected.

### shim2018

**Yes:**
- Noiseless Cube
- Noiseless AFAContext
- Noiseless synthetic MNIST (if increasing model complexity)

---

### zannone2019

**Yes:**
- Noiseless Cube
- Noiseless synthetic MNIST (910be785354e)
- Noiseless AFAContext (910be785354e)

### kachuee2019

**Yes:**
- Noiseless synthetic MNIST
- Noiseless Cube (maybe, but needs a bit larger patience, debugging this)

**No:**
- Noiseless AFAContext

Since kachuee2019 is not able to distinguish between missing features and features with values 0, these results are not surprising.

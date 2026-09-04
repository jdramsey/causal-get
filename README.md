# Causal-GET
Causal Graph Estimation Toolbox

---

## How to build and install:

```
python -m build
pip install dist/*.whl
```

---

## Example usage:

```python
import numpy as np
import pandas as pd

import daosim as ds
import causalget as cg


## SIMULATION PARAMETERS ##

n = 1000  # NUMBER OF SAMPLES
p = 100   # NUMBER OF VARIABLES
ad = 10   # AVERAGE DEGREE

## SIMULATION DATA VIA DAO ##

g = ds.er_dag(p, ad=ad)
_, B, O = ds.corr(g)
X = ds.simulate(B, O, n)
df = pd.DataFrame(X)
R = df.corr().values

## TESTING FROM COV ##

print("from ndarray (corr):")
dag = cg.boss(R, n=n, discount=2.0, restarts=10)
print("SHD:", np.sum(dag != g))
print()

## TESTING FROM DATA ##

print("from datafrom (data):")
dag = cg.boss(df, discount=2.0, restarts=1)
print("SHD:", np.sum(dag != g))
print()
```

## Nonlinear data: BOSS with the basis-function BIC (BF-BIC)

`cg.boss` scores with the linear Gaussian BIC and is unchanged. For continuous data with nonlinear
dependencies, `cg.boss_bf` runs the same search with Tetrad's `BasisFunctionBicScore`: each variable
is expanded into a block of Legendre basis columns (`causalget.embedding`, a port of Tetrad's
`Embedding.getEmbeddedData`), and the local score of a variable given its parents is the joint
Gaussian BIC of its block given the union of the parents' blocks, on the correlation matrix of the
embedded data. The search itself (GST, BOSS, restarts, knowledge) is identical; only the scorer sees
the blocks.

```python
dag = cg.boss_bf(X, truncation_limit=3, discount=2.0, restarts=1, seed=1)   # X: n x p ndarray or DataFrame
```

`truncation_limit` is Tetrad's TRUNCATION_LIMIT, `discount` its PENALTY_DISCOUNT (2 is the BF-BIC
default), `lam` its SINGULARITY_LAMBDA, `rank_transform` its BASIS_RANK_TRANSFORM. BF-BIC needs the raw data, not a covariance matrix. Scores
are not comparable across `boss` and `boss_bf`.

`cg.local_score(R, n, y, parents, discount, offsets, lam)` exposes the C scorer in Tetrad BIC units
for checking against Tetrad; see `tests/bf_bic_vs_tetrad.py`.

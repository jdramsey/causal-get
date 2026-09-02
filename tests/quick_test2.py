import time

import numpy as np
import pandas as pd

import daosim as ds
import causalget as cg


# dag[y, x] == 1 means x is a parent of y (row = child, columns = parents).
# This matches what causalget.boss returns; flip PARENTS_ARE_COLUMNS if daosim's
# er_dag uses the other convention and the true-graph BIC comes out worse than
# every estimate.
PARENTS_ARE_COLUMNS = True


def bic(dag, R, n, discount):
  """Total BIC of a DAG given a correlation matrix, in the usual maximand form:

      sum_y  -n * log(resid_var_y)  -  discount * k_y * log(n)

  Scored in float64 regardless of what the search used, so this is an
  independent referee rather than the search's own opinion of itself.
  """
  p = R.shape[0]
  logn = np.log(n)
  total = 0.0

  for y in range(p):
    z = np.nonzero(dag[y] if PARENTS_ARE_COLUMNS else dag[:, y])[0]
    k = len(z)

    if k == 0:
      resid = R[y, y]
    else:
      Rzz = R[np.ix_(z, z)]
      Rzy = R[z, y]
      try:
        resid = R[y, y] - Rzy @ np.linalg.solve(Rzz, Rzy)
      except np.linalg.LinAlgError:
        resid = R[y, y] - Rzy @ np.linalg.lstsq(Rzz, Rzy, rcond=None)[0]

    if not np.isfinite(resid) or resid <= 0:
      return np.nan  # singular parent set -- see the note about p >> n below

    total += -n * np.log(resid) - discount * k * logn

  return total


def report(name, dag, g, R, n, discount, ms, true_bic):
  # Cellwise disagreement, not textbook SHD: a reversed edge counts 2, a
  # missing or extra edge counts 1. Fine for comparing arms to each other.
  diff = int(np.sum(dag != g))
  score = bic(dag, R, n, discount)
  dbic = score - true_bic
  print(f"  {name:<18} edges={int(dag.sum()):6d}  diff={diff:7d}  "
        f"dBIC={dbic:12.1f}  {ms:8.0f} ms")


if __name__ == "__main__":
  n = 500
  p = 2000
  ad = 6

  discount = 2
  restarts = 1
  seed = 29

  # tol is the minimum improvement, in BIC points, that better_mutation will
  # accept. It used to be hard-coded at 1e-3 in per-node score units, which is
  # 2 * n * 1e-3 BIC points -- 1.0 here at n=500, 10.0 at n=5000. The 1e0 arm
  # below is therefore roughly the old behaviour at this sample size.
  tols = [1e0, 1e-1, 1e-2, 1e-4]

  print(f"p={p} n={n} ad={ad} discount={discount} restarts={restarts} seed={seed}")

  g = ds.er_dag(p, ad=ad)
  _, B, O = ds.cov(g)
  X = ds.simulate(B, O, n)
  df = pd.DataFrame(X)
  R = df.corr().values

  true_bic = bic(g, R, n, discount)
  print(f"true graph: edges={int(g.sum())}  BIC={true_bic:.1f}")
  if not np.isfinite(true_bic):
    print("  true BIC is not finite -- check PARENTS_ARE_COLUMNS, or p >> n is "
          "making some parent set singular")
  print()

  ## TOLERANCE SWEEP (one build, one process) ##

  print("from ndarray (corr):")
  for tol in tols:
    t = time.time()
    dag = cg.boss(R, n=n, discount=discount, restarts=restarts, seed=seed, tol=tol)
    ms = (time.time() - t) * 1000
    report(f"tol={tol:g}", dag, g, R, n, discount, ms, true_bic)
  print()

  ## TESTING FROM DATA ##

  # print("from ndarray (data):")
  # t = time.time()
  # dag = cg.boss(X, discount=discount, restarts=restarts, seed=seed)
  # report("data", dag, g, R, n, discount, (time.time() - t) * 1000, true_bic)
  # print()

  # print("from dataframe:")
  # t = time.time()
  # dag = cg.boss(df, discount=discount, restarts=restarts, seed=seed)
  # report("dataframe", dag, g, R, n, discount, (time.time() - t) * 1000, true_bic)
  # print()

  ## TESTING SEEDS ##

  # dag1 = cg.boss(R, n=n, discount=discount, seed=32)
  # dag2 = cg.boss(R, n=n, discount=discount, seed=32)
  # dag3 = cg.boss(R, n=n, discount=discount, seed=23)

  # print(f"seed test: {int(np.sum(dag1 == dag2))} / {p * p}")
  # print(f"seed test: {int(np.sum(dag2 == dag3))} / {p * p}")

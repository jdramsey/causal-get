"""Basis-function embedding for BF-BIC, ported from Tetrad's Embedding.getEmbeddedData
(tetrad-lib/src/main/java/edu/cmu/tetrad/search/utils/Embedding.java, development branch,
basisType = 1 (Legendre), basisScale = 1) so that C-BOSS with `offsets` scores the same
BasisFunctionBicScore that Tetrad does.

Only continuous variables are handled. Tetrad also embeds discrete variables as indicator blocks;
that is a small, separate addition and is not done here.

For each column x:
  1. rescale to [-1, 1]: by default min-max (DataTransforms.scale with -basisScale, basisScale);
     with rank_transform=True, mid-ranks mapped to -1 + 2 (k + 0.5) / n with ties averaged
     (Embedding.rankTransformToUnitInterval, Tetrad's BASIS_RANK_TRANSFORM), so a column with m
     distinct values keeps m distinct values and its rank-drop behaviour;
  2. raw basis columns P_1(x), ..., P_T(x), Legendre polynomials via the three-term recurrence;
  3. numerical-rank pruning: drop P_k (k > 1) when its residual after projecting out the
     intercept and the previously KEPT columns is <= 1e-8 * its centered norm (a binary column
     keeps only its linear term, a c-valued column at most c - 1 terms);
  4. the kept columns are orthonormalized by modified Gram-Schmidt (two passes, NOT centered),
     in low-order-first order, so the first column of every block spans the linear term.

Step 4 is an invertible linear map within the block and therefore does not change any BF-BIC
score DIFFERENCE (see the notes in bic.h); it is reproduced so that absolute local scores agree
with Tetrad's as well. Only step 3, the drop decision, affects the search.
"""

import numpy as np


def legendre_columns(x, truncation_limit):
  """P_1(x) .. P_T(x) as columns of an (n, T) array, via the Legendre recurrence
  k P_k = (2k - 1) x P_{k-1} - (k - 1) P_{k-2}, matching StatUtils.legendre."""
  n = x.shape[0]
  out = np.empty((n, truncation_limit))
  p_prev = np.ones(n)
  p_cur = x.copy()
  out[:, 0] = p_cur
  for k in range(2, truncation_limit + 1):
    p_next = ((2 * k - 1) * x * p_cur - (k - 1) * p_prev) / k
    out[:, k - 1] = p_next
    p_prev, p_cur = p_cur, p_next
  return out


def _scale_to_unit(x):
  lo = x.min()
  hi = x.max()
  if hi == lo:
    # DataTransforms.scale divides by (max - min); a constant column is degenerate for Tetrad
    # too. Map it to zeros so it embeds as a (dropped-by-rank) constant rather than NaN.
    return np.zeros_like(x)
  return -1.0 + 2.0 * (x - lo) / (hi - lo)


def _rank_to_unit(x):
  """Mid-ranks to a uniform grid on [-1, 1], ties averaged (Embedding.rankTransformToUnitInterval)."""
  n = x.shape[0]
  order = np.argsort(x, kind="stable")
  xs = x[order]
  out = np.empty(n)
  k = 0
  while k < n:
    m = k
    while m + 1 < n and xs[m + 1] == xs[k]:
      m += 1
    mid = 0.5 * (k + m)
    out[order[k:m + 1]] = -1.0 + 2.0 * (mid + 0.5) / n
    k = m + 1
  return out


def embed_block(x, truncation_limit, rel_tol=1e-8, rank_transform=False):
  """Embeds one continuous column. Returns (kept_columns (n, k), kept_orders list)."""
  n = x.shape[0]
  x = np.asarray(x, dtype=np.float64)
  x = _rank_to_unit(x) if rank_transform else _scale_to_unit(x)
  raw = legendre_columns(x, truncation_limit)

  if truncation_limit == 1:
    return raw[:, :1].copy(), [1]

  # Drop-test basis: normalized intercept plus the orthonormalized kept directions.
  drop_basis = [np.full(n, 1.0 / np.sqrt(n))]
  kept_stored = []
  kept_orders = []

  for order in range(1, truncation_limit + 1):
    col = raw[:, order - 1]

    centered_norm = np.linalg.norm(col - col.mean())

    resid = col.copy()
    for _ in range(2):
      for u in drop_basis:
        resid -= (u @ resid) * u
    resid_norm = np.linalg.norm(resid)

    dependent = resid_norm <= rel_tol * max(centered_norm, 1e-12)
    if order > 1 and dependent:
      continue

    if resid_norm > 0.0:
      drop_basis.append(resid / resid_norm)

    q = col.copy()
    for _ in range(2):
      for prev in kept_stored:
        q -= (prev @ q) * prev
    q_norm = np.linalg.norm(q)
    if q_norm > 0.0:
      q /= q_norm

    kept_stored.append(q)
    kept_orders.append(order)

  return np.column_stack(kept_stored), kept_orders


def embed(X, truncation_limit=3, rel_tol=1e-8, rank_transform=False):
  """Embeds every column of X (n, p).

  rank_transform: rank-transform each column to [-1, 1] before the Legendre expansion instead of
  min-max scaling (Tetrad's BASIS_RANK_TRANSFORM). Gives every row the same leverage; see the Tetrad
  Javadoc on Embedding.RANK_TRANSFORM for why this is recommended.

  Returns
  -------
  X_emb : (n, m) float64 array of embedded columns, variable blocks contiguous and in order
  offsets : (p + 1,) uint32 array; variable v owns columns offsets[v] .. offsets[v+1] - 1
  orders : list of lists, the Legendre orders kept for each variable
  """
  X = np.asarray(X, dtype=np.float64)
  if X.ndim != 2:
    raise ValueError("X must be 2-d (n, p)")
  if truncation_limit < 1:
    raise ValueError("truncation_limit must be >= 1")
  if not np.isfinite(X).all():
    raise ValueError("data contains NaN or inf")

  blocks = []
  orders = []
  offsets = [0]
  # Apple's Accelerate BLAS raises spurious floating-point exception flags through numpy's matmul
  # (reported as divide-by-zero / overflow in matmul on perfectly finite data), so warnings are
  # suppressed here and finiteness is checked explicitly below instead.
  with np.errstate(all="ignore"):
    for j in range(X.shape[1]):
      b, o = embed_block(X[:, j], truncation_limit, rel_tol, rank_transform)
      blocks.append(b)
      orders.append(o)
      offsets.append(offsets[-1] + b.shape[1])

  X_emb = np.hstack(blocks)
  if not np.isfinite(X_emb).all():
    raise ValueError("embedding produced non-finite values")
  return X_emb, np.asarray(offsets, dtype=np.uint32), orders


def embedded_correlation(X, truncation_limit=3, rel_tol=1e-8, rank_transform=False):
  """Convenience: the correlation matrix of the embedding (what Tetrad's BasisFunctionBicScore
  hands to SemBicScore), plus offsets and kept orders."""
  X_emb, offsets, orders = embed(X, truncation_limit, rel_tol, rank_transform)
  R = np.corrcoef(X_emb, rowvar=False)
  return R, offsets, orders

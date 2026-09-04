import sys
import struct

import numpy as np
import pandas as pd
import threading

from .c_backend import (
  boss_from_cov,
  boss_from_data,
  local_score as _local_score,
)
from .embedding import embed, embedded_correlation


def _kwargs(discount, restarts, seed, tol, offsets=None, lam=0.0):
  kw = dict(discount=float(discount), restarts=int(restarts), tol=float(tol))
  if seed is not None: kw["seed"] = int(seed)
  if offsets is not None:
    kw["offsets"] = offsets
    kw["lambda"] = float(lam)
  return kw

def worker_bfc(cov_buf, knwl_buf, discount, restarts, seed, tol, ret, offsets=None, lam=0.0):
  ret["blob"] = boss_from_cov(cov_buf, knwl_buf, **_kwargs(discount, restarts, seed, tol, offsets, lam))

def worker_bfd(data_buf, knwl_buf, discount, restarts, seed, tol, ret):
  ret["blob"] = boss_from_data(data_buf, knwl_buf, **_kwargs(discount, restarts, seed, tol))


def _pack_knowledge(knowledge, names, forbid_within, byte_order):
  """Serializes tiered knowledge for the C backend.

  knowledge: dict mapping a tier number to the list of variables in that tier. Tiers are ordered by
  their keys; a variable in tier k may take parents only from tiers <= k. The tiers MUST partition the
  variables: every variable in exactly one tier. (Tetrad lets an unlisted variable float freely across
  tiers; a suborder search cannot express that, so it is an error here rather than a silent choice.)
  Variables may be given as column names (DataFrame input) or 0-based indices (ndarray input).

  forbid_within: set of tier numbers inside which no edges are allowed (Tetrad's starred tier).

  Buffer layout, all uint32: num_groups, group_sizes[num_groups], group_members[p] in group order,
  num_forbidden, then (group_a, group_b, type) triples where type bit 1 forbids a -> b and bit 2 forbids
  b -> a, so a starred tier k is the triple (k, k, 3).
  """
  index = {name: i for i, name in enumerate(names)}
  p = len(names)
  if not knowledge:
    return struct.pack(byte_order + "III", 0, 0, 0)

  tiers = sorted(knowledge.keys())
  seen = {}
  groups = []
  for t in tiers:
    members = []
    for v in knowledge[t]:
      if v not in index:
        raise ValueError(f"knowledge refers to unknown variable {v!r}")
      if v in seen:
        raise ValueError(f"variable {v!r} appears in tiers {seen[v]} and {t}; tiers must partition the variables")
      seen[v] = t
      members.append(index[v])
    if not members:
      raise ValueError(f"tier {t} is empty")
    groups.append(members)
  missing = [name for name in names if name not in seen]
  if missing:
    raise ValueError(f"{len(missing)} variable(s) are in no tier, e.g. {missing[:5]}; tiers must partition "
                     f"the variables (put unconstrained variables in their own tier)")

  forbidden = []
  for t in (forbid_within or ()):
    if t not in knowledge:
      raise ValueError(f"forbid_within names tier {t}, which is not in knowledge")
    forbidden += [tiers.index(t), tiers.index(t), 3]

  buf = struct.pack(byte_order + "I", len(groups))
  buf += struct.pack(byte_order + f"{len(groups)}I", *[len(g) for g in groups])
  buf += struct.pack(byte_order + f"{p}I", *[i for g in groups for i in g])
  buf += struct.pack(byte_order + "I", len(forbidden) // 3)
  if forbidden:
    buf += struct.pack(byte_order + f"{len(forbidden)}I", *forbidden)
  return buf


def boss(data, n=None, discount=1.0, restarts=1, knowledge=None, seed=None, tol=1e-2, forbid_within=None):
  '''
  Runs the Best Order Score Serch (BOSS).

  Parameters
  ----------
  data = covariance matrix or dataset (ndarray / datafrome)
  n = specifies the number of samples (only set if passing a covariance matrix) 
  discount = specifies the penalty discount for the BIC score
  restarts = speficies the number of random restarts
  knowledge = dict mapping a tier number to the list of variables in that tier (column names for a
               DataFrame, 0-based indices for an ndarray). Tiers must partition the variables; a
               variable in tier k may have parents only in tiers <= k.
  forbid_within = set of tier numbers inside which no edges are allowed (Tetrad's starred tier)
  seed = used to set the random seed
  tol = min improvement, in BIC points, for a transposition to be accepted

  Returns
  -------
  g = direct acyclic graph
  '''

  byte_order = "<" if sys.byteorder == "little" else ">"

  if isinstance(data, pd.DataFrame):
    names = list(data.columns)
  elif isinstance(data, np.ndarray):
    names = list(range(data.shape[1]))
  else:
    raise TypeError("data must be a pandas DataFrame or a numpy ndarray")
  knwl_buf = _pack_knowledge(knowledge, names, forbid_within, byte_order)


  ret = {}

  if isinstance(n, int) and isinstance(data, np.ndarray):
    _, p = data.shape
    if not np.isfinite(data).all():
      raise ValueError("correlation matrix contains NaN or inf")
    R = np.ascontiguousarray(data, dtype=np.float64)
    cov_buf = struct.pack(byte_order + "II", n, p)
    cov_buf += R.tobytes()
    thread = threading.Thread(target=worker_bfc, args=(cov_buf, knwl_buf, discount, restarts, seed, tol, ret)) 

  elif isinstance(data, np.ndarray):
    n, p = data.shape
    X = np.ascontiguousarray(data.T, dtype=np.float64) # transposed: variable-major
    data_buf = struct.pack(byte_order + "II", n, p)
    data_buf += X.tobytes()
    thread = threading.Thread(target=worker_bfd, args=(data_buf, knwl_buf, discount, restarts, seed, tol, ret)) 

  elif isinstance(data, pd.DataFrame):
    # n and p were never assigned on this branch, so boss(DataFrame) raised UnboundLocalError
    n, p = data.shape
    if not np.isfinite(data.values).all():
      raise ValueError("data contains NaN or inf")
    R = np.ascontiguousarray(data.corr().values, dtype=np.float64)
    cov_buf = struct.pack(byte_order + "II", n, p)
    cov_buf += R.tobytes()
    thread = threading.Thread(target=worker_bfc, args=(cov_buf, knwl_buf, discount, restarts, seed, tol, ret)) 
    # print("boss from data")
    # n, p = data.shape
    # X = data.astype(np.float32).values.T # float32 transposed
    # data_buf = struct.pack(byte_order + "II", n, p)
    # data_buf += X.tobytes()
    # thread = threading.Thread(target=worker_bfd, args=(cov_buf, knwl_buf, discount, restarts, seed, tol, ret)) 

  else:
    # replace with raise
    print("ERROR: invalid input")
    exit(1)

  thread.start()

  try:
    while thread.is_alive():
      thread.join(timeout=0.1)
  except KeyboardInterrupt:
    # replace with raise
    print("Interrupted")
    exit(1)

  blob = ret["blob"]

  STRUCT_FMT = byte_order + "iii"
  STRUCT_SIZE = struct.calcsize(STRUCT_FMT)
  edges = [struct.unpack_from(STRUCT_FMT, blob, offset) for offset in range(0, len(blob), STRUCT_SIZE)]

  dag = np.zeros([p, p], dtype=np.uint8)
  for i, j, e in edges:
    if e == 2: dag[i, j] = 1
    if e == 1: dag[j, i] = 1

  return dag


def _run(thread):
  thread.start()
  try:
    while thread.is_alive():
      thread.join(timeout=0.1)
  except KeyboardInterrupt:
    raise


def _unpack_dag(blob, p, byte_order):
  STRUCT_FMT = byte_order + "iii"
  STRUCT_SIZE = struct.calcsize(STRUCT_FMT)
  edges = [struct.unpack_from(STRUCT_FMT, blob, offset) for offset in range(0, len(blob), STRUCT_SIZE)]
  dag = np.zeros([p, p], dtype=np.uint8)
  for i, j, e in edges:
    if e == 2: dag[i, j] = 1
    if e == 1: dag[j, i] = 1
  return dag


def _pack_cov(R, n, byte_order):
  R = np.ascontiguousarray(R, dtype=np.float64)
  m = R.shape[0]
  return struct.pack(byte_order + "II", int(n), int(m)) + R.tobytes()


def _pack_offsets(offsets, byte_order):
  return struct.pack(byte_order + f"{len(offsets)}I", *[int(o) for o in offsets])


def boss_bf(data, truncation_limit=3, discount=2.0, restarts=1, knowledge=None, seed=None,
            tol=1e-2, forbid_within=None, lam=0.0, rank_transform=False, return_embedding=False):
  '''
  Runs BOSS with the basis-function BIC (BF-BIC) score, for nonlinear (non-Gaussian-linear)
  continuous data. Mirrors Tetrad's BasisFunctionBicScore: each variable is expanded into a
  block of Legendre basis columns (see causalget.embedding), and the local score of a variable
  given its parents is the joint linear-Gaussian BIC of the child's block given the union of
  the parents' blocks, computed on the correlation matrix of the embedded data.

  Parameters
  ----------
  data = dataset (ndarray / DataFrame), n rows by p continuous columns. A covariance matrix
         cannot be used here: the embedding needs the raw data.
  truncation_limit = highest Legendre order per variable (Tetrad's TRUNCATION_LIMIT; default 3)
  discount = penalty discount (Tetrad's BF-BIC default is 2)
  lam = singularity lambda, a ridge on the regressor block (Tetrad's SINGULARITY_LAMBDA)
  rank_transform = rank-transform columns to [-1, 1] before embedding instead of min-max scaling
                   (Tetrad's BASIS_RANK_TRANSFORM; recommended by Tetrad, off by default to match it)
  restarts, knowledge, seed, tol, forbid_within = as in boss()
  return_embedding = also return (offsets, kept_orders)

  Returns
  -------
  g = directed acyclic graph over the ORIGINAL p variables (and optionally the embedding info)
  '''
  byte_order = "<" if sys.byteorder == "little" else ">"

  if isinstance(data, pd.DataFrame):
    names = list(data.columns)
    X = data.values
  elif isinstance(data, np.ndarray):
    names = list(range(data.shape[1]))
    X = data
  else:
    raise TypeError("data must be a pandas DataFrame or a numpy ndarray")

  n, p = X.shape
  knwl_buf = _pack_knowledge(knowledge, names, forbid_within, byte_order)

  R, offsets, orders = embedded_correlation(X, truncation_limit, rank_transform=rank_transform)
  if not np.isfinite(R).all():
    raise ValueError("embedded correlation matrix contains NaN or inf")

  cov_buf = _pack_cov(R, n, byte_order)
  offs_buf = _pack_offsets(offsets, byte_order)

  ret = {}
  thread = threading.Thread(target=worker_bfc,
                            args=(cov_buf, knwl_buf, discount, restarts, seed, tol, ret, offs_buf, lam))
  _run(thread)

  dag = _unpack_dag(ret["blob"], p, byte_order)
  if return_embedding:
    return dag, offsets, orders
  return dag


def local_score(R, n, y, parents, discount=2.0, offsets=None, lam=0.0):
  '''
  Local score of variable y given `parents`, in Tetrad BIC units (2 * sum(lik) - c * dof * log n,
  with Tetrad's SemBicScore likelihood constants), computed by the C scorer. With `offsets` this is
  BF-BIC on an embedded correlation matrix R (see embedded_correlation); without, plain SEM BIC on R.
  Intended for checking the C scorer against Tetrad's BasisFunctionBicScore / SemBicScore.
  '''
  byte_order = "<" if sys.byteorder == "little" else ">"
  cov_buf = _pack_cov(R, n, byte_order)
  par_buf = struct.pack(byte_order + f"{len(parents)}I", *[int(v) for v in parents])
  kw = dict(discount=float(discount))
  if offsets is not None:
    kw["offsets"] = _pack_offsets(offsets, byte_order)
    kw["lambda"] = float(lam)
  return _local_score(cov_buf, int(y), par_buf, **kw)

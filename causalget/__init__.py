import sys
import struct

import numpy as np
import pandas as pd
import threading

from .c_backend import (
  boss_from_cov,
  boss_from_data,
)


def _kwargs(discount, restarts, seed, tol):
  kw = dict(discount=float(discount), restarts=int(restarts), tol=float(tol))
  if seed is not None: kw["seed"] = int(seed)
  return kw

def worker_bfc(cov_buf, knwl_buf, discount, restarts, seed, tol, ret):
  ret["blob"] = boss_from_cov(cov_buf, knwl_buf, **_kwargs(discount, restarts, seed, tol))

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
    R = data.astype(np.float32) # float32
    cov_buf = struct.pack(byte_order + "II", n, p)
    cov_buf += R.tobytes()
    thread = threading.Thread(target=worker_bfc, args=(cov_buf, knwl_buf, discount, restarts, seed, tol, ret)) 

  elif isinstance(data, np.ndarray):
    n, p = data.shape
    X = data.astype(np.float32).T # float32 transposed 
    data_buf = struct.pack(byte_order + "II", n, p)
    data_buf += X.tobytes()
    thread = threading.Thread(target=worker_bfd, args=(data_buf, knwl_buf, discount, restarts, seed, tol, ret)) 

  elif isinstance(data, pd.DataFrame):
    # n and p were never assigned on this branch, so boss(DataFrame) raised UnboundLocalError
    n, p = data.shape
    if not np.isfinite(data.values).all():
      raise ValueError("data contains NaN or inf")
    R = data.corr().astype(np.float32).values # float32
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

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


# currently ignoring knowledge
def boss(data, n=None, discount=1.0, restarts=1, knowledge=None, seed=None, tol=1e-2):
  '''
  Runs the Best Order Score Serch (BOSS).

  Parameters
  ----------
  data = covariance matrix or dataset (ndarray / datafrome)
  n = specifies the number of samples (only set if passing a covariance matrix) 
  discount = specifies the penalty discount for the BIC score
  restarts = speficies the number of random restarts
  knowledge = dictionary mapping uints (zero is forbid within) to list of strings
  seed = used to set the random seed
  tol = min improvement, in BIC points, for a transposition to be accepted

  Returns
  -------
  g = direct acyclic graph
  '''

  byte_order = "<" if sys.byteorder == "little" else ">"



  # for tier in knowledge:



  knwl_buf = struct.pack(byte_order + "III", 0, 0, 0)


  ret = {}

  if isinstance(n, int) and isinstance(data, np.ndarray):
    _, p = data.shape
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

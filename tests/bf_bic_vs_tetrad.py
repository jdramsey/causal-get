"""
Checks C-BOSS's BF-BIC against Tetrad's BasisFunctionBicScore, two ways:

  1. LOCAL SCORES. For random (child, parent-set) pairs, compare
       cg.local_score(R_emb, n, y, parents, offsets=...)        (C scorer, Tetrad BIC units)
     against
       new BasisFunctionBicScore(data, T, lambda, false, rankTransform).localScore(y, parents)
     (branch joe-work-2026-8-24 API; falls back to older constructors if the jar lacks it).
     They should agree to ~1e-6 relative (the C side holds the correlation matrix in float32).
     If they do NOT, the script also reports which of two numpy references Tetrad's numbers match:
     the chain-rule (joint-likelihood) score the C code implements, or the pre-2026-8 diagonal-
     residual form -- a mismatch of that kind means the jar predates the working branch.

  2. SEARCH. Run cg.boss_bf and Tetrad's Boss on the same BF-BIC score and compare the CPDAGs.
     Tetrad's Boss returns a CPDAG; the C DAG is converted with GraphTransforms.dagToCpdag. Ties
     between equivalent DAGs, and BOSS's own restart/tie behaviour, mean the DAGs need not be
     identical, but the CPDAGs should be identical or nearly so on the same score.

Usage:
    python tests/bf_bic_vs_tetrad.py [--p 15] [--n 1000] [--T 3] [--lam 0.0] [--checks 60] [--rank-transform]

Requires: causalget (this repo, built), pytetrad with resources/tetrad-current.jar built from
tetrad branch joe-work-2026-8-24, jpype1. Uses adaptiveBasisSelection=false on the Tetrad side
(adaptive pruning is not ported to the C side).
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--p", type=int, default=15)
parser.add_argument("--n", type=int, default=1000)
parser.add_argument("--T", type=int, default=3, help="truncation limit")
parser.add_argument("--lam", type=float, default=0.0, help="singularity lambda")
parser.add_argument("--discount", type=float, default=2.0)
parser.add_argument("--checks", type=int, default=60, help="number of random local-score checks")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--rank-transform", action="store_true", help="Tetrad's BASIS_RANK_TRANSFORM")
parser.add_argument("--xmx", type=str, default="4g")
parser.add_argument("--jar", type=str, default=None,
                    help="path to tetrad-current.jar; default is the one inside the INSTALLED pytetrad package, "
                         "which may not be your py-tetrad checkout")
args = parser.parse_args()

import causalget as cg
from causalget.embedding import embedded_correlation

import jpype
import jpype.imports
import importlib.resources as importlib_resources

jar_path = args.jar or str(importlib_resources.files("pytetrad").joinpath("resources", "tetrad-current.jar"))
if not os.path.exists(jar_path):
    sys.exit(f"tetrad-current.jar not found at {jar_path}")
if jpype.isJVMStarted():
    sys.exit("JVM already started before this script could set the classpath")
jpype.startJVM(jpype.getDefaultJVMPath(), f"-Xmx{args.xmx}", classpath=[jar_path])
print(f"jar: {jar_path}")
print(f"     modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(jar_path)))}, "
      f"{os.path.getsize(jar_path) / 1e6:.1f} MB")

# Fingerprint the loaded BasisFunctionBicScore so a stale jar is obvious.
import jpype.imports  # noqa: E402  (JVM must be running before edu.* imports)
from java.lang import Class as JClass
_cls = JClass.forName("edu.cmu.tetrad.search.score.BasisFunctionBicScore")
_methods = {str(m.getName()) for m in _cls.getMethods()}
_ctor_arity = sorted(len(c.getParameterTypes()) for c in _cls.getConstructors())
print(f"     BasisFunctionBicScore constructors (arg counts): {_ctor_arity}; "
      f"getEmbedding: {'getEmbedding' in _methods}; embeddingBlockSizes: {'embeddingBlockSizes' in _methods}")
if 'embeddingBlockSizes' not in _methods or 5 not in _ctor_arity:
    print("     WARNING: this is NOT the joe-work-2026-8-24 BasisFunctionBicScore (expected a 5-arg constructor and\n"
          "     embeddingBlockSizes()). Pass --jar /path/to/your/fresh/tetrad-current.jar, or reinstall py-tetrad\n"
          "     from your checkout (pip install -e /path/to/py-tetrad).")

import pytetrad.tools.translate as tr
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.search.score as tscore
from edu.cmu.tetrad.graph import GraphTransforms
from jpype import JArray, JInt

# --------------------------------------------------------------------------------------
# Nonlinear additive-noise data on a random DAG
# --------------------------------------------------------------------------------------

rng = np.random.default_rng(args.seed)
n, p = args.n, args.p
order = rng.permutation(p)
X = np.zeros((n, p))
funcs = [np.tanh, lambda x: np.clip(x, -3, 3) ** 2, np.sin, lambda x: x]  # bounded, so chains cannot blow up
for k, j in enumerate(order):
    pa = [i for i in order[:k] if rng.random() < 3.0 / max(p - 1, 1)]
    X[:, j] = rng.standard_normal(n)
    for i in pa:
        f = funcs[rng.integers(len(funcs))]
        X[:, j] += rng.uniform(0.5, 1.5) * rng.choice([-1, 1]) * f(X[:, i])
X[:, 0] = np.round(X[:, 0])  # a few-valued column, to exercise the rank-drop in the embedding
if not np.isfinite(X).all():
    sys.exit("simulated data is not finite")
names = [f"X{i + 1}" for i in range(p)]  # 1-based, to match tr.adj_matrix_to_graph
df = pd.DataFrame(X, columns=names)

# --------------------------------------------------------------------------------------
# 1. Local scores
# --------------------------------------------------------------------------------------

data = tr.pandas_data_to_tetrad(df)
try:
    # joe-work-2026-8-24: (DataSet, truncationLimit, lambda, adaptiveBasisSelection, rankTransform)
    score = tscore.BasisFunctionBicScore(data, args.T, args.lam, False, args.rank_transform)
except TypeError:
    if args.rank_transform:
        sys.exit("this jar has no rankTransform constructor; rebuild tetrad-current.jar from joe-work-2026-8-24")
    try:
        score = tscore.BasisFunctionBicScore(data, args.T, args.lam, False)
    except TypeError:
        print("NOTE: jar has only the 3-arg constructor; it predates the 2026-8 BF-BIC changes")
        score = tscore.BasisFunctionBicScore(data, args.T, args.lam)
score.setPenaltyDiscount(args.discount)

R, offsets, orders = embedded_correlation(X, args.T, rank_transform=args.rank_transform)
c_sizes = [int(offsets[i + 1] - offsets[i]) for i in range(p)]
tet_sizes = None
if 'embeddingBlockSizes' in _methods:
    tet_sizes = [int(v) for v in score.embeddingBlockSizes()]
elif 'getEmbedding' in _methods:
    tet_emb = score.getEmbedding()
    tet_sizes = [len(tet_emb.get(JInt(i))) for i in range(p)]
print(f"embedded block sizes  Tetrad: {tet_sizes if tet_sizes is not None else '(not exposed by this jar)'}")
print(f"                      C     : {c_sizes}")
if tet_sizes is not None and tet_sizes != c_sizes:
    print("WARNING: block sizes differ; the rank-drop decisions do not match Tetrad's")


def ref_score(y, pa, chain_rule):
    """numpy BF-BIC in Tetrad units. chain_rule=True is the working-branch (joint-likelihood) form
    the C code implements; False is the pre-2026-8 diagonal-residual form."""
    A = list(range(offsets[y], offsets[y + 1]))
    B = [j for v in pa for j in range(offsets[v], offsets[v + 1])]
    lik, dof = 0.0, 0
    for a in A:
        if B:
            # SemBicScore.getResidualVariance: ridge coefficients, residual under the UNregularized cov
            b = np.linalg.solve(R[np.ix_(B, B)] + args.lam * np.eye(len(B)), R[B, a])
            s2 = R[a, a] - 2 * b @ R[B, a] + b @ R[np.ix_(B, B)] @ b
        else:
            s2 = R[a, a]
        lik += -0.5 * n * (np.log(2 * np.pi * s2) + 1)
        dof += len(B)
        if chain_rule:
            B = B + [a]
    return 2 * lik - args.discount * dof * np.log(n)


worst = worst_chain = worst_diag = 0.0
for _ in range(args.checks):
    y = int(rng.integers(p))
    k = int(rng.integers(0, 5))
    pa = [int(v) for v in rng.choice([v for v in range(p) if v != y], size=k, replace=False)]
    tet = float(score.localScore(y, JArray(JInt)(pa)))
    c = cg.local_score(R, n, y, pa, discount=args.discount, offsets=offsets, lam=args.lam)
    rel = abs(tet - c) / max(1.0, abs(tet))
    if not np.isfinite(rel):
        rel = np.inf
    worst = max(worst, rel)
    worst_chain = max(worst_chain, abs(tet - ref_score(y, pa, True)) / max(1.0, abs(tet)))
    worst_diag = max(worst_diag, abs(tet - ref_score(y, pa, False)) / max(1.0, abs(tet)))
    if rel > 1e-5:
        print(f"  mismatch: y={y} pa={pa}  Tetrad {tet:.6f}  C {c:.6f}")
print(f"local scores: {args.checks} random (child, parents) checks, worst relative discrepancy C vs Tetrad {worst:.2e}")
if worst > 1e-5:
    print(f"  diagnostics: Tetrad vs numpy chain-rule reference {worst_chain:.2e}; vs diagonal-residual reference {worst_diag:.2e}")
    if worst_diag < 1e-6 < worst_chain:
        print("  -> Tetrad is computing the OLD diagonal-residual score: this jar predates the working branch.")

# --------------------------------------------------------------------------------------
# 2. Search
# --------------------------------------------------------------------------------------

t = time.time()
dag_c = cg.boss_bf(X, truncation_limit=args.T, discount=args.discount, lam=args.lam,
                   rank_transform=args.rank_transform, seed=args.seed)
t_c = time.time() - t

t = time.time()
boss = ts.Boss(score)
boss.setNumStarts(1)
boss.setUseBes(False)
boss.setUseDataOrder(True)
cpdag_t = ts.PermutationSearch(boss).search()
t_t = time.time() - t

# C convention: dag_c[child, parent] == 1
adj = np.zeros((p, p), dtype=int)
for ch, pa in zip(*np.nonzero(dag_c)):
    adj[pa, ch] = 1
dag_c_java = tr.adj_matrix_to_graph(adj)
cpdag_c = GraphTransforms.dagToCpdag(dag_c_java)


def edge_set(g):
    out = set()
    for e in g.getEdges():
        a, b = str(e.getNode1().getName()), str(e.getNode2().getName())
        out.add((a, b, str(e.getEndpoint1()), str(e.getEndpoint2())))
        out.add((b, a, str(e.getEndpoint2()), str(e.getEndpoint1())))
    return out


ec, et = edge_set(cpdag_c), edge_set(cpdag_t)
print(f"search: C-BOSS(BF-BIC) {t_c:.2f}s, Tetrad Boss(BF-BIC) {t_t:.2f}s")
print(f"        C CPDAG edges {cpdag_c.getNumEdges()}, Tetrad CPDAG edges {cpdag_t.getNumEdges()}, "
      f"differing (directed-edge records) {len(ec ^ et) // 2}")
if ec != et:
    print("        edges only in C:      ", sorted({e for e in ec - et if e[0] < e[1]}))
    print("        edges only in Tetrad: ", sorted({e for e in et - ec if e[0] < e[1]}))

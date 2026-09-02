"""
compare_four.py -- a four-arm comparison of linear Gaussian permutation searches.

Arms
----
  TET-FLOP   edu.cmu.tetrad.search.Flop           (Java, via py-tetrad / tetrad-current.jar)
  TET-BOSS   edu.cmu.tetrad.search.Boss           (Java, via py-tetrad / tetrad-current.jar)
  RUST-FLOP  flopsearch.flop                      (Rust, the FLOP authors' reference implementation)
  C-BOSS     causalget.boss                       (C, Bryan Andrews' causal-get)

Everything is simulated once per replication and handed to all four arms, so the four see
identical data. All four are scored afterwards by ONE referee -- Tetrad's SemBicScore at
the same penalty discount -- so the accuracy columns are commensurable even though the
four implementations compute their own internal BICs differently.

Requires Python 3.12 (the causal-get wheel is cp312-only).

Simulation
----------
Two generators, selected with --sim:

  --sim daosim   (default) Bryan's daosim: er_dag -> corr -> simulate. The model is built
                 directly in correlation scale, so every variable has unit variance by
                 construction and there is nothing for a variance-sorting heuristic to
                 exploit. The cost is that the implied edge coefficients are small -- mean
                 |coef| around 0.16 at ad=6 -- so recall at n=1000 is low (~0.55 at p=100,
                 ~0.26 at p=500), and the recovered graphs are much sparser than the truth.
 -
  --sim sem      Tetrad's SemIm: RandomGraph.randomGraph -> SemPm -> SemIm -> simulateData.
                 Coefficients are drawn from +/-[--coef-low, --coef-high] (Tetrad's default
                 is [0, 1], mean |coef| 0.5) and error variances from [--var-low, --var-high]
                 (default [1, 3]). Stronger signal, higher recall, denser recoveries -- and
                 therefore a much harder timing test for the grow-shrink-tree searches,
                 whose cost is driven by parent-set size rather than by p.

On varsortability: every arm here is fed either the correlation matrix or column-
standardized data, so the marginal-variance ordering that SemIm data carries is removed
before any search sees it. What survives standardization is R^2-sortability (Reisach et
al. 2023), which SemIm data does exhibit and daosim data does not. So `--sim sem` is not a
varsortability leak in this harness, but it is not a fully neutral generator either;
report which one you used.

Install
-------
    python3.12 -m pip install numpy pandas daosim jpype1 flopsearch
    python3.12 -m pip install git+https://github.com/cmu-phil/py-tetrad

    # causal-get, built from the fork rather than the Google Drive wheel:
    git clone https://github.com/jdramsey/causal-get && cd causal-get
    rm -rf build dist *.egg-info
    python3.12 -m build --wheel
    python3.12 -m pip install --force-reinstall dist/*.whl

Upstream HEAD (bja43/causal-get) does not compile and its boss_from_cov runs sp_search
from the identity order while ignoring `restarts`; the fork fixes both. The `rm -rf build`
is not optional -- setuptools caches c_backend.o and keys the cache on source mtime, not
on compile flags. flopsearch has a cp312 macosx_11_0_arm64 wheel on PyPI, so plain pip
install works for that one.

IMPORTANT: make sure the causalget you install lands in the SAME interpreter that runs
this script. Multiple python3.12 installations on one machine is the failure mode to
watch for; the script prints causalget.__file__ and its signature at startup so a
mismatch is visible rather than silent.

Run
---
    python3.12 compare_four.py                                  # daosim, p=100, ad=6, n=1000
    python3.12 compare_four.py --sim sem                        # Tetrad SemIm, default ranges
    python3.12 compare_four.py --sim sem --coef-low 0.2 --coef-high 0.7
    python3.12 compare_four.py --p 500 --ad 6 --reps 5
    python3.12 compare_four.py --arms TET-FLOP,RUST-FLOP --p 1000 --reps 3
    python3.12 compare_four.py --boss-threads 4                 # BOSS multithreaded (see caveats)

Columns
-------
    ms      wall-clock milliseconds for the search call only
    dBIC    referee BIC of the returned graph minus referee BIC of the true DAG.
            Tetrad convention: higher is better. Positive means the arm found a graph
            the score likes better than the truth (normal at these sample sizes).
    shd     structural Hamming distance, estimated CPDAG vs. CPDAG of the true DAG
    AP/AR   adjacency precision / recall vs. the true CPDAG
    AHP/AHR arrowhead precision / recall vs. the true CPDAG

Caveats worth keeping in mind when you read the table
-----------------------------------------------------
1. THREADS. Tetrad BOSS parallelizes over a ForkJoinPool; the other three arms are
   single-threaded. The default here is --boss-threads 1 so the times are algorithmic
   rather than a core count. Set it to 4 if you want the "what a user actually gets"
   number, but then the ms column is no longer like-for-like.

2. RESTART SEMANTICS DIFFER, and the script normalizes them. --searches is the number
   of LOCAL SEARCHES each arm performs. The mappings were read off the sources:
     Tetrad BOSS   numStarts   = searches      (PermutationSearch runs numStarts searches)
     Tetrad FLOP   numRestarts = searches - 1  (restarts are additional searches)
     flopsearch    restarts    = searches - 1  (algo.rs: limit = restarts + 1)
     causal-get    restarts    = searches      (boss.h: boss_restarts loops r in 0..restarts,
                                                with r == 0 running from the incoming order)
   The two FLOPs use ILS restarts (perturb the incumbent); the two BOSSes use random
   restarts unless you pass --boss-ils. If you want the closest algorithmic pairing,
   compare TET-FLOP against RUST-FLOP, and TET-BOSS (no --boss-ils) against C-BOSS.

2a. SEEDING IS NOT UNIFORM. Tetrad FLOP, Tetrad BOSS and C-BOSS all take a seed and this
   script passes seed + rep to each. flopsearch does NOT: its Rust code uses ThreadRng and
   flop(data, lambda_bic, *, restarts, timeout) has no seed parameter. So RUST-FLOP is the
   one arm that is not reproducible run to run, and small changes in its row across runs
   are noise. Do not read a few dBIC points of movement there as a real effect.

   (Historical note, since it invalidated several earlier runs: run_c_boss used to probe
   the wheel's signature by calling it inside `try: ... except TypeError: continue`, and
   the call passed `restarts` twice. That raised TypeError unconditionally, the handler
   swallowed it, and the probe fell through to a key set with no seed -- so C-BOSS was
   silently unseeded too. It now reads inspect.signature instead of probing by exception.)

2b. PRECISION. causal-get casts the correlation matrix to float32 before scoring; the
   other three arms score in double precision. The cast is a single ~1e-7 relative
   perturbation of an input whose sampling error is ~1/sqrt(n), so it is well below noise;
   sweeping the fork's `tol` across four orders of magnitude moved the score not at all.
   Retained here as a difference worth knowing about, not as an explanation for anything.

2c. TETRAD BOSS RESETS ITS TREES. Boss.resetAfterRS defaults to true, so each restart
   rebuilds every grow-shrink tree from scratch; C-BOSS builds them once and keeps them
   warm across restarts. --boss-no-reset makes the two comparable on that axis. Measured
   at p=500, ad=6, n=1000 with reset on: TET-BOSS 370 s against C-BOSS 1 s at parity dBIC.

3. INPUT ASYMMETRY. Tetrad FLOP, Tetrad BOSS and C-BOSS are all handed the p x p
   correlation matrix. flopsearch's Python API only accepts the raw n x p data matrix,
   so RUST-FLOP pays an extra O(n p^2) correlation pass inside its timed region. The
   script prints how long that pass takes in numpy for the same data, so you can judge
   how much of the RUST-FLOP time is that rather than search.

4. JIT. The JVM is warmed up on a throwaway problem before any timing, so the Java arms
   are not paying first-call interpretation costs. JVM startup itself is outside all
   timed regions.

5. TIMING DEPENDS ON THE GENERATOR. Grow-shrink-tree search cost is governed by how large
   the parent sets the search FINDS are, not by p. On daosim data at n=1000 the recovered
   graphs are sparse, the trees saturate within a few restarts, and the search is mostly
   cache lookup. On SemIm data recall is much higher, the trees are deeper and wider, and
   every arm slows down -- the GST arms most. The same build measured 40x apart on two
   generators at identical p and n. Always report --sim alongside the ms column.
"""

import argparse
import csv
import inspect
import os
import sys
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Argument parsing (before the JVM starts, so --help is fast)
# --------------------------------------------------------------------------------------

ALL_ARMS = ["TET-FLOP", "TET-BOSS", "RUST-FLOP", "C-BOSS"]

parser = argparse.ArgumentParser(description="Four-arm linear Gaussian search comparison.")
parser.add_argument("--p", type=int, default=100, help="number of variables")
parser.add_argument("--ad", type=float, default=6.0, help="average degree")
parser.add_argument("--n", type=int, default=1000, help="sample size")
parser.add_argument("--reps", type=int, default=10, help="number of replications")
parser.add_argument("--searches", type=int, default=21,
                    help="number of local searches per arm (see caveat 2)")
parser.add_argument("--penalty", type=float, default=2.0,
                    help="BIC penalty discount / lambda, shared by all arms and the referee")
parser.add_argument("--sim", choices=["daosim", "sem"], default="daosim",
                    help="data generator: daosim (correlation scale) or sem (Tetrad SemIm)")
parser.add_argument("--coef-low", type=float, default=0.0,
                    help="--sim sem: lower bound of |coef|; Tetrad's default is 0.0")
parser.add_argument("--coef-high", type=float, default=1.0,
                    help="--sim sem: upper bound of |coef|; Tetrad's default is 1.0")
parser.add_argument("--var-low", type=float, default=1.0,
                    help="--sim sem: lower bound of error variance; Tetrad's default is 1.0")
parser.add_argument("--var-high", type=float, default=3.0,
                    help="--sim sem: upper bound of error variance; Tetrad's default is 3.0")
parser.add_argument("--boss-threads", type=int, default=1,
                    help="threads for Tetrad BOSS; 1 keeps the ms column like-for-like")
parser.add_argument("--boss-ils", action="store_true",
                    help="use ILS restarts in Tetrad BOSS instead of random restarts")
parser.add_argument("--boss-no-reset", action="store_true",
                    help="setResetAfterRS(false): keep the grow-shrink tree score cache warm "
                         "across restarts, as causal-get's C BOSS does. Costs memory, since "
                         "the GSTs then grow monotonically over the whole run.")
parser.add_argument("--cboss-tol", type=float, default=None,
                    help="causal-get's acceptance tolerance in BIC points; only supported "
                         "by builds from the fork. Omit to use the wheel's default.")
parser.add_argument("--arms", type=str, default=",".join(ALL_ARMS),
                    help="comma-separated subset of " + ",".join(ALL_ARMS))
parser.add_argument("--seed", type=int, default=29,
                    help="master seed; replication r uses seed + r")
parser.add_argument("--xmx", type=str, default="8g", help="JVM max heap, e.g. 8g")
parser.add_argument("--csv", type=str, default="compare_four.csv", help="output CSV path")
parser.add_argument("--append", action="store_true",
                    help="append to the CSV instead of overwriting; useful for sweeping "
                         "--searches with a fixed --seed to get accuracy-vs-time curves")
args = parser.parse_args()

ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]
for a in ARMS:
    if a not in ALL_ARMS:
        sys.exit(f"Unknown arm {a!r}; choose from {ALL_ARMS}")

# --------------------------------------------------------------------------------------
# JVM startup. Must happen before any edu.cmu.tetrad imports.
# --------------------------------------------------------------------------------------

import jpype
import jpype.imports
import importlib.resources as importlib_resources

jar_path = str(importlib_resources.files("pytetrad").joinpath("resources", "tetrad-current.jar"))
if not os.path.exists(jar_path):
    sys.exit(f"tetrad-current.jar not found at {jar_path}")

if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), f"-Xmx{args.xmx}", classpath=[jar_path])

import java.lang as jlang
import java.util as jutil
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.search.score as tscore
import edu.cmu.tetrad.search.utils as tsutils
import edu.cmu.tetrad.sem as tsem
import edu.cmu.tetrad.util as tutil

from edu.cmu.tetrad.graph import GraphTransforms, RandomGraph
from edu.cmu.tetrad.search.utils import GraphSearchUtils

# --------------------------------------------------------------------------------------
# Optional third-party arms and generators
# --------------------------------------------------------------------------------------

ds = None
if args.sim == "daosim":
    try:
        import daosim as ds
    except ImportError:
        sys.exit("daosim not installed: python3.12 -m pip install daosim  (or use --sim sem)")

flopsearch = None
if "RUST-FLOP" in ARMS:
    try:
        import flopsearch
    except ImportError:
        sys.exit("flopsearch not installed: python3.12 -m pip install flopsearch")

cg = None
if "C-BOSS" in ARMS:
    try:
        import causalget as cg
    except ImportError:
        sys.exit("causalget not installed; build it from the fork (see docstring)")


# --------------------------------------------------------------------------------------
# Graph conversions.
#
# Two matrix conventions are in play and they are transposes of one another:
#
#   daosim / causal-get:  A[i][j] == 1  means  j -> i     (row = child, col = parent)
#   flopsearch:           M[i][j] == 1  means  i -> j,
#                         M[i][j] == M[j][i] == 2 means i -- j (undirected)
#
# Both generators below return the true DAG in the first convention.
#
# Note also that py-tetrad's translate.adj_matrix_to_graph documents the first convention
# in its comment but implements the second in its body -- worth fixing in py-tetrad; this
# script does its own conversions rather than relying on it.
# --------------------------------------------------------------------------------------

def make_nodes(p):
    """One shared list of Tetrad nodes, X1..Xp, used by every graph and by the score."""
    nodes = jutil.ArrayList()
    for i in range(1, p + 1):
        nodes.add(td.ContinuousVariable(f"X{i}"))
    return nodes


def dag_matrix_to_graph(A, nodes):
    """A[i][j] == 1 means j -> i (the daosim / causal-get convention)."""
    g = tg.EdgeListGraph(nodes)
    idx = np.argwhere(np.asarray(A) != 0)
    for i, j in idx:
        g.addDirectedEdge(nodes.get(int(j)), nodes.get(int(i)))
    return g


def flop_matrix_to_graph(M, nodes):
    """M[i][j] == 1 means i -> j; a pair of 2's means an undirected edge."""
    M = np.asarray(M)
    p = M.shape[0]
    g = tg.EdgeListGraph(nodes)
    for i in range(p):
        for j in range(i + 1, p):
            a, b = M[i, j], M[j, i]
            if a == 2 or b == 2:
                g.addUndirectedEdge(nodes.get(i), nodes.get(j))
            elif a == 1:
                g.addDirectedEdge(nodes.get(i), nodes.get(j))
            elif b == 1:
                g.addDirectedEdge(nodes.get(j), nodes.get(i))
    return g


def normalize_to_cpdag(g):
    """Put every arm's output in the same representation before scoring.

    causal-get returns a DAG; Tetrad BOSS/FLOP and flopsearch return CPDAGs. Comparing a
    DAG against the true CPDAG counts every compelled-vs-reversible distinction as an
    arrowhead error, which unfairly deflates arrowhead precision for the DAG-returning
    arm. Applying dagToCpdag to any legal DAG fixes that, and is a no-op in effect for an
    arm that already returned a CPDAG containing undirected edges."""
    return GraphTransforms.dagToCpdag(g) if g.paths().isLegalDag() else g


def graph_to_arrays(g, p, name_to_index):
    """Return (adj, arrow): adj is symmetric; arrow[i][j] == 1 iff the i-j edge has an
    arrowhead at j (i.e. i -> j)."""
    adj = np.zeros((p, p), dtype=bool)
    arrow = np.zeros((p, p), dtype=bool)
    for e in g.getEdges():
        i = name_to_index[str(e.getNode1().getName())]
        j = name_to_index[str(e.getNode2().getName())]
        adj[i, j] = adj[j, i] = True
        if str(e.getEndpoint2().name()) == "ARROW":
            arrow[i, j] = True
        if str(e.getEndpoint1().name()) == "ARROW":
            arrow[j, i] = True
    return adj, arrow


# --------------------------------------------------------------------------------------
# Data generators. Each returns (A, X): A is the true DAG in the daosim convention,
# X is the n x p data matrix with columns in X1..Xp order. Both are deterministic in seed.
# --------------------------------------------------------------------------------------

def simulate_daosim(p, ad, n, seed):
    rng = np.random.default_rng(seed)
    g = ds.er_dag(p, ad=ad, rng=rng)
    _, B, O = ds.corr(g, rng=rng)
    X = ds.simulate(B, O, n, rng=rng)
    return np.asarray(g, dtype=int), np.asarray(X, dtype=np.float64)


def simulate_sem(p, ad, n, seed, coef_low, coef_high, var_low, var_high):
    """Tetrad's own simulation: random DAG with p*ad/2 edges, SemPm, SemIm with the given
    coefficient and error-variance ranges, then simulateData. Coefficients are drawn
    from +/-[coef_low, coef_high] (SemIm's Split distribution with coefSymmetric=true).

    APIs checked against the development branch of cmu-phil/tetrad:
      RandomGraph.randomGraph(List<Node>, int latents, int maxEdges, int maxDegree,
                              int maxIndegree, int maxOutdegree, boolean connected, long seed)
      SemIm(SemPm, Parameters)  reading "coefLow", "coefHigh", "varLow", "varHigh"
      SemIm.simulateData(int, boolean) -> DataSet
    """
    nodes = make_nodes(p)
    num_edges = int(round(p * ad / 2.0))
    no_cap = 1000  # effectively uncapped degree, so ER-style degree spread is preserved

    tutil.RandomUtil.getInstance().setSeed(jpype.JLong(seed))
    graph = RandomGraph.randomGraph(nodes, 0, num_edges, no_cap, no_cap, no_cap, False,
                                    jpype.JLong(seed))

    params = tutil.Parameters()
    params.set("coefLow", jpype.JDouble(coef_low))
    params.set("coefHigh", jpype.JDouble(coef_high))
    params.set("varLow", jpype.JDouble(var_low))
    params.set("varHigh", jpype.JDouble(var_high))

    tutil.RandomUtil.getInstance().setSeed(jpype.JLong(seed))
    im = tsem.SemIm(tsem.SemPm(graph), params)
    data = im.simulateData(n, False)

    # simulateData does not promise X1..Xp column order, so reorder by name.
    names = [str(s) for s in data.getVariableNames()]
    X = np.array(data.getDoubleData().toArray(), dtype=np.float64)
    X = X[:, [names.index(f"X{i + 1}") for i in range(p)]]

    idx = {f"X{i + 1}": i for i in range(p)}
    A = np.zeros((p, p), dtype=int)
    for node in graph.getNodes():
        child = idx[str(node.getName())]
        for parent in graph.getParents(node):
            A[child, idx[str(parent.getName())]] = 1
    return A, X


def simulate(p, ad, n, seed):
    if args.sim == "daosim":
        return simulate_daosim(p, ad, n, seed)
    return simulate_sem(p, ad, n, seed, args.coef_low, args.coef_high,
                        args.var_low, args.var_high)


# --------------------------------------------------------------------------------------
# Referee: one SemBicScore for all arms, mirroring bic() in BossIlsFlopStudy.java.
# --------------------------------------------------------------------------------------

def referee_bic(graph, score, name_to_index):
    """Common-scorer BIC, Tetrad convention (higher is better). If the graph is a CPDAG,
    a DAG is drawn from its equivalence class first; BIC is score-equivalent, so which
    member is drawn does not matter."""
    dag = graph if graph.paths().isLegalDag() else GraphTransforms.dagFromCpdag(graph)
    total = 0.0
    for node in dag.getNodes():
        parents = dag.getParents(node)
        pa = jpype.JArray(jpype.JInt)(
            [name_to_index[str(parents.get(k).getName())] for k in range(parents.size())])
        total += float(score.localScore(name_to_index[str(node.getName())], pa))
    return total


def metrics(est_graph, true_cpdag_arrays, true_dag_graph, score, p, name_to_index, true_bic):
    # dBIC is computed from the original graph (BIC needs a DAG); the structural columns
    # are computed from the CPDAG so all four arms are compared like for like.
    dbic = referee_bic(est_graph, score, name_to_index) - true_bic
    est_graph = normalize_to_cpdag(est_graph)
    adj_t, arr_t = true_cpdag_arrays
    adj_e, arr_e = graph_to_arrays(est_graph, p, name_to_index)

    def pr(est, tru):
        tp = int(np.sum(est & tru))
        return (tp / max(int(np.sum(est)), 1), tp / max(int(np.sum(tru)), 1))

    ap, ar = pr(np.triu(adj_e), np.triu(adj_t))
    ahp, ahr = pr(arr_e, arr_t)
    shd = int(GraphSearchUtils.structuralhammingdistance(true_dag_graph, est_graph, True))
    return dict(shd=shd, dBIC=dbic, AP=ap, AR=ar, AHP=ahp, AHR=ahr)


# --------------------------------------------------------------------------------------
# The four arms. Each returns (est_graph, elapsed_ms).
# --------------------------------------------------------------------------------------

def to_java_2d(M):
    M = np.ascontiguousarray(M, dtype=np.float64)
    try:
        return jpype.JArray.of(M)
    except Exception:
        return jpype.JArray(jpype.JDouble, 2)(M.tolist())


def tetrad_cov(R, n, nodes):
    """Tetrad ICovarianceMatrix wrapping the correlation matrix."""
    return td.CovarianceMatrix(nodes, to_java_2d(R), n)


def run_tetrad_flop(cov, penalty, searches, seed):
    flop = ts.Flop(cov)
    flop.setPenaltyDiscount(penalty)
    flop.setNumRestarts(max(searches - 1, 0))
    flop.setSeed(seed)
    t0 = time.perf_counter()
    est = flop.search()
    return est, (time.perf_counter() - t0) * 1000.0


def used_heap_mb():
    """Rough live-heap reading. System.gc() is advisory, so treat this as a differential
    measurement (reset vs. no-reset on the same problem), not an absolute footprint."""
    rt = jlang.Runtime.getRuntime()
    for _ in range(2):
        jlang.System.gc()
    return (float(rt.totalMemory()) - float(rt.freeMemory())) / (1024.0 * 1024.0)


def run_tetrad_boss(cov, penalty, searches, seed, threads, ils, no_reset=False):
    score = tscore.SemBicScore(cov, penalty)
    boss = ts.Boss(score)
    boss.setNumStarts(searches)
    boss.setNumThreads(threads)
    boss.setUseBes(False)
    if no_reset:
        boss.setResetAfterRS(False)
    try:
        boss.setUseIlsRestarts(ils)
    except AttributeError:
        if ils:
            raise RuntimeError("this tetrad-current.jar has no Boss.setUseIlsRestarts; "
                               "rebuild the jar or drop --boss-ils")
    ps = ts.PermutationSearch(boss)
    ps.setSeed(seed)
    t0 = time.perf_counter()
    est = ps.search()
    ms = (time.perf_counter() - t0) * 1000.0
    # Read the heap while boss/ps (and therefore the grow-shrink trees) are still reachable.
    heap = used_heap_mb()
    del ps, boss, score
    return est, ms, heap


def run_rust_flop(X_std, penalty, searches, nodes):
    t0 = time.perf_counter()
    M = flopsearch.flop(X_std, float(penalty), restarts=max(searches - 1, 0))
    ms = (time.perf_counter() - t0) * 1000.0
    return flop_matrix_to_graph(np.asarray(M), nodes), ms


_cboss_params = None  # cached parameter names of the installed causalget.boss


def cboss_signature():
    """Read the installed wheel's signature once, rather than probing by exception.

    The earlier version called cg.boss(R, restarts=searches, **kw) where kw could also
    contain 'restarts'. That raises TypeError for a duplicate keyword on every wheel,
    the bare `except TypeError: continue` swallowed it, and the probe fell through to a
    key set carrying no seed -- so C-BOSS ran unseeded while reporting that the wheel
    did not support seeding. Introspection cannot fail that way."""
    global _cboss_params
    if _cboss_params is None:
        _cboss_params = set(inspect.signature(cg.boss).parameters)
        print(f"causalget: {cg.__file__}")
        print(f"           boss{inspect.signature(cg.boss)}")
        for name, consequence in (
                ("restarts", "C-BOSS runs at its built-in default budget, not --searches"),
                ("seed", "C-BOSS is not seed-matched to the other arms"),
        ):
            if name not in _cboss_params:
                print(f"           note: no {name!r} argument; {consequence}")
        if "tol" not in _cboss_params:
            print("           note: no 'tol' argument, so this wheel predates the cov-path "
                  "fix; boss_from_cov runs sp_search from the identity order and ignores "
                  "restarts internally, whatever this script passes")
    return _cboss_params


def run_c_boss(R, n, penalty, searches, seed, nodes, tol=None):
    """causalget.boss returns a DAG in the A[i][j] == 1 means j -> i convention.

    causal-get's boss_restarts loops `for (r = 0; r < restarts; r++)`, so restarts is the
    number of local searches, not the number of extra ones."""
    params = cboss_signature()

    kw = dict(n=n, discount=penalty)
    if "restarts" in params:
        kw["restarts"] = searches
    if "seed" in params:
        kw["seed"] = seed
    if tol is not None and "tol" in params:
        kw["tol"] = tol

    R = np.ascontiguousarray(R, dtype=np.float64)
    t0 = time.perf_counter()
    A = cg.boss(R, **kw)          # no try/except: a TypeError here is a real bug
    ms = (time.perf_counter() - t0) * 1000.0

    return dag_matrix_to_graph(np.asarray(A), nodes), ms


# --------------------------------------------------------------------------------------
# Convention self-test. Cheap insurance against a silent transpose.
# --------------------------------------------------------------------------------------

def self_test():
    nodes = make_nodes(3)
    # Chain 0 -> 1 -> 2 in the daosim convention.
    A = np.zeros((3, 3), dtype=int)
    A[1, 0] = 1
    A[2, 1] = 1
    dag = dag_matrix_to_graph(A, nodes)
    assert dag.isParentOf(nodes.get(0), nodes.get(1)), "dag_matrix_to_graph transposed"
    cpdag = GraphTransforms.dagToCpdag(dag)

    # The same chain's CPDAG in the flopsearch convention: X1 -- X2 -- X3.
    M = np.zeros((3, 3), dtype=int)
    M[0, 1] = M[1, 0] = 2
    M[1, 2] = M[2, 1] = 2
    other = flop_matrix_to_graph(M, nodes)
    shd = int(GraphSearchUtils.structuralhammingdistance(cpdag, other, False))
    assert shd == 0, f"flop_matrix_to_graph disagrees with dagToCpdag (shd={shd})"

    # A v-structure must survive as oriented in both directions of conversion.
    B = np.zeros((3, 3), dtype=int)
    B[2, 0] = 1
    B[2, 1] = 1
    v = GraphTransforms.dagToCpdag(dag_matrix_to_graph(B, nodes))
    N = np.zeros((3, 3), dtype=int)
    N[0, 2] = 1
    N[1, 2] = 1
    assert int(GraphSearchUtils.structuralhammingdistance(
        v, flop_matrix_to_graph(N, nodes), False)) == 0, "v-structure conversion mismatch"

    # The selected generator must return an acyclic graph in the daosim convention and a
    # data matrix of the right shape, reproducibly.
    A1, X1 = simulate(12, 3.0, 50, 7)
    A2, X2 = simulate(12, 3.0, 50, 7)
    assert A1.shape == (12, 12) and X1.shape == (50, 12), "simulate() shape mismatch"
    assert np.array_equal(A1, A2) and np.allclose(X1, X2), "simulate() not seed-reproducible"
    assert dag_matrix_to_graph(A1, make_nodes(12)).paths().isLegalDag(), \
        "simulate() returned a cyclic graph"
    print(f"conventions self-test: ok  (sim={args.sim})")


def warm_up_jvm(penalty):
    """Run both Java arms once on a small throwaway problem so the timed runs are JITted."""
    _, X = simulate(20, 4.0, 500, 0)
    R = np.corrcoef(X, rowvar=False)
    nodes = make_nodes(20)
    cov = tetrad_cov(R, 500, nodes)
    for _ in range(3):
        run_tetrad_flop(cov, penalty, 5, 1)
        run_tetrad_boss(cov, penalty, 5, 1, args.boss_threads, args.boss_ils,
                        args.boss_no_reset)
    print("JVM warm-up: done")


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------

def main():
    self_test()
    if "C-BOSS" in ARMS:
        cboss_signature()   # report the installed build before anything is timed
    warm_up_jvm(args.penalty)

    p, n, ad = args.p, args.n, args.ad
    nodes = make_nodes(p)
    name_to_index = {f"X{i + 1}": i for i in range(p)}

    sim_desc = args.sim
    if args.sim == "sem":
        sim_desc += (f" (coef +/-[{args.coef_low:g}, {args.coef_high:g}], "
                     f"var [{args.var_low:g}, {args.var_high:g}])")

    print()
    print(f"p={p}  avg degree={ad}  n={n}  reps={args.reps}  searches/arm={args.searches}  "
          f"penalty={args.penalty}  boss threads={args.boss_threads}  "
          f"boss ils={args.boss_ils}  boss reset-after-restart="
          f"{not args.boss_no_reset}")
    print(f"sim: {sim_desc}")
    print(f"arms: {', '.join(ARMS)}")
    print()

    fields = ["rep", "arm", "sim", "p", "ad", "n", "searches", "penalty",
              "ms", "dBIC", "shd", "AP", "AR", "AHP", "AHR", "heap_mb"]
    exists = args.append and os.path.exists(args.csv) and os.path.getsize(args.csv) > 0
    csv_file = open(args.csv, "a" if args.append else "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    if not exists:
        writer.writeheader()

    totals = {a: dict(ms=0.0, dBIC=0.0, shd=0.0, AP=0.0, AR=0.0, AHP=0.0, AHR=0.0, k=0)
              for a in ARMS}

    for rep in range(args.reps):
        seed = args.seed + rep

        # ---- simulate once; every arm sees exactly this data ----
        g, X = simulate(p, ad, n, seed)

        t0 = time.perf_counter()
        R = np.corrcoef(X, rowvar=False)
        corr_ms = (time.perf_counter() - t0) * 1000.0
        X_std = np.ascontiguousarray((X - X.mean(0)) / X.std(0), dtype=np.float64)

        # ---- referee setup ----
        cov = tetrad_cov(R, n, nodes)
        ref_score = tscore.SemBicScore(cov, args.penalty)
        true_dag = dag_matrix_to_graph(g, nodes)
        true_bic = referee_bic(true_dag, ref_score, name_to_index)
        true_cpdag = GraphTransforms.dagToCpdag(true_dag)
        true_arrays = graph_to_arrays(true_cpdag, p, name_to_index)

        print(f"rep {rep + 1}/{args.reps}  (seed {seed}, |E|={int(np.sum(g))}, "
              f"numpy corrcoef took {corr_ms:.0f} ms)")

        for arm in ARMS:
            heap = float("nan")
            try:
                if arm == "TET-FLOP":
                    est, ms = run_tetrad_flop(cov, args.penalty, args.searches, seed)
                elif arm == "TET-BOSS":
                    est, ms, heap = run_tetrad_boss(cov, args.penalty, args.searches, seed,
                                                    args.boss_threads, args.boss_ils,
                                                    args.boss_no_reset)
                elif arm == "RUST-FLOP":
                    est, ms = run_rust_flop(X_std, args.penalty, args.searches, nodes)
                elif arm == "C-BOSS":
                    est, ms = run_c_boss(R, n, args.penalty, args.searches, seed, nodes,
                                         tol=args.cboss_tol)
                else:
                    continue

                m = metrics(est, true_arrays, true_dag, ref_score, p, name_to_index, true_bic)
            except Exception as e:  # keep the run alive; one broken arm should not cost the rest
                print(f"   {arm:10s}  FAILED: {type(e).__name__}: {e}")
                continue

            row = dict(rep=rep + 1, arm=arm, sim=args.sim, p=p, ad=ad, n=n,
                       searches=args.searches, penalty=args.penalty, ms=round(ms, 1),
                       dBIC=round(m["dBIC"], 1), shd=m["shd"], AP=round(m["AP"], 4),
                       AR=round(m["AR"], 4), AHP=round(m["AHP"], 4), AHR=round(m["AHR"], 4),
                       heap_mb=("" if heap != heap else round(heap, 1)))
            writer.writerow(row)
            csv_file.flush()

            t = totals[arm]
            for key in ("ms", "dBIC", "shd", "AP", "AR", "AHP", "AHR"):
                t[key] += float(row[key])
            if heap == heap:
                t["heap"] = t.get("heap", 0.0) + heap
                t["heap_k"] = t.get("heap_k", 0) + 1
            t["k"] += 1

            heap_str = "" if heap != heap else f"  heap={heap:7.1f}MB"
            print(f"   {arm:10s}  ms={ms:9.0f}  dBIC={m['dBIC']:9.1f}  shd={m['shd']:5d}  "
                  f"AP={m['AP']:.3f} AR={m['AR']:.3f}  "
                  f"AHP={m['AHP']:.3f} AHR={m['AHR']:.3f}{heap_str}")
        sys.stdout.flush()

    csv_file.close()

    print()
    print(f"averages over completed replications  (sim={sim_desc})")
    print(f"{'arm':10s} {'reps':>5s} {'ms':>10s} {'dBIC':>10s} {'shd':>8s} "
          f"{'AP':>6s} {'AR':>6s} {'AHP':>6s} {'AHR':>6s}")
    for arm in ARMS:
        t = totals[arm]
        if t["k"] == 0:
            print(f"{arm:10s} {0:5d}   (no completed replications)")
            continue
        k = t["k"]
        hk = t.get("heap_k", 0)
        heap_str = f" {t['heap'] / hk:9.1f}MB" if hk else ""
        print(f"{arm:10s} {k:5d} {t['ms'] / k:10.0f} {t['dBIC'] / k:10.1f} "
              f"{t['shd'] / k:8.1f} {t['AP'] / k:6.3f} {t['AR'] / k:6.3f} "
              f"{t['AHP'] / k:6.3f} {t['AHR'] / k:6.3f}{heap_str}")
    print()
    print(f"per-replication rows written to {args.csv}")


if __name__ == "__main__":
    main()
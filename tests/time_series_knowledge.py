"""
time_series_knowledge.py -- exercise C-BOSS's tiered knowledge on lagged time-series data.

Pipeline
--------
  1. Tetrad builds a random DAG over p variables and expands it into a TimeLagGraph with `numLags`
     lags (TsUtils.graphToLagGraph), then draws SEM coefficients for it (SemIm).
  2. We read those coefficients back and simulate a genuine time series of length T from the
     structural VAR they define: X_t = B0 X_t + sum_k Bk X_{t-k} + e_t, solved for X_t at each step.
     (TimeSeriesSemSimulation draws i.i.d. rows from the lag graph instead; a real series is what the
     user actually has, and it carries the autocorrelation that matters for sample-size arguments.)
  3. TsUtils.createLagData turns the series into the lagged data set X:0..X:numLags with Tetrad's own
     tiered knowledge: lag numLags in the first tier, ..., lag 0 last. Every lagged variable is in
     exactly one tier, which is the partition C-BOSS requires.
  4. The lagged data and the tiers go to causalget.boss. Optionally Tetrad's BOSS runs on the same
     data with the same knowledge as a reference arm.
  5. Accuracy is scored on edges INTO lag-0 variables only, which is the part of a lag graph that is
     identifiable and the part a forecaster cares about: adjacency and arrowhead precision/recall
     against the true lag graph, with contemporaneous (lag-0 -> lag-0) edges treated separately since
     tiers do not orient them.

Run from the tests directory:
    python time_series_knowledge.py                        # p=10, 2 lags, T=2000
    python time_series_knowledge.py --p 20 --lags 3 --T 5000 --tetrad-boss
    python time_series_knowledge.py --star                 # forbid contemporaneous edges (starred lag-0 tier)

Requires jpype1 and a Tetrad jar: uses py-tetrad's tetrad-current.jar if py-tetrad is installed,
else ./tetrad-current.jar in this directory.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--p", type=int, default=10, help="number of time-series variables")
parser.add_argument("--lags", type=int, default=2, help="number of lags in the true graph and in the lagged data")
parser.add_argument("--T", type=int, default=2000, help="length of the simulated series (after burn-in)")
parser.add_argument("--ad", type=float, default=2.0, help="average degree of the base DAG before lag expansion")
parser.add_argument("--coef-low", type=float, default=0.2)
parser.add_argument("--coef-high", type=float, default=0.6)
parser.add_argument("--penalty", type=float, default=2.0)
parser.add_argument("--restarts", type=int, default=5)
parser.add_argument("--seed", type=int, default=29)
parser.add_argument("--star", action="store_true", help="forbid edges within the lag-0 tier (no contemporaneous edges)")
parser.add_argument("--tetrad-boss", action="store_true", help="also run Tetrad's BOSS with the same knowledge as a reference")
parser.add_argument("--no-knowledge", action="store_true", help="also run C-BOSS with no tiers, to see what the knowledge buys")
args = parser.parse_args()

# --------------------------------------------------------------------------------------
# JVM
# --------------------------------------------------------------------------------------

import jpype
import jpype.imports

jar = None
try:
    import importlib.resources as ir
    cand = str(ir.files("pytetrad").joinpath("resources", "tetrad-current.jar"))
    if os.path.exists(cand):
        jar = cand
except Exception:
    pass
if jar is None:
    jar = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tetrad-current.jar")
if not os.path.exists(jar):
    sys.exit("no tetrad-current.jar found")

jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx4g", classpath=[jar])

import java.util as jutil
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.search.score as tscore
import edu.cmu.tetrad.sem as tsem
import edu.cmu.tetrad.util as tutil
from edu.cmu.tetrad.graph import RandomGraph
from edu.cmu.tetrad.search.utils import TsUtils

import causalget as cg

# --------------------------------------------------------------------------------------
# 1. Random lag graph with SEM coefficients
# --------------------------------------------------------------------------------------

p, L = args.p, args.lags
tutil.RandomUtil.getInstance().setSeed(jpype.JLong(args.seed))

nodes = jutil.ArrayList()
for i in range(1, p + 1):
    nodes.add(td.ContinuousVariable(f"X{i}"))
base = RandomGraph.randomGraph(nodes, 0, int(round(p * args.ad / 2)), 100, 100, 100, False, jpype.JLong(args.seed))
# graphToLagGraph adds X:1 -> X:0 for every variable and replicates the base DAG at lag 0. Its
# `extraLaggedEdges` argument draws the extra cross-lag edges with an unseeded Collections.shuffle, so
# we ask for none and add them here from a seeded numpy generator to keep the run reproducible.
lag_graph = TsUtils.graphToLagGraph(base, L, 0)
extra_lagged = int(np.floor(base.getNumEdges() * 1.5))
rng_graph = np.random.default_rng(args.seed + 1)
pairs = [(i, j) for i in range(p) for j in range(p)]
rng_graph.shuffle(pairs)
added = 0
for i, j in pairs:
    if added >= extra_lagged:
        break
    lag = int(rng_graph.integers(1, L + 1))
    frm = lag_graph.getNode(f"X{i + 1}", lag)
    to = lag_graph.getNode(f"X{j + 1}", 0)
    if lag_graph.isAdjacentTo(frm, to):
        continue
    lag_graph.addEdge(tg.Edge(frm, to, tg.Endpoint.TAIL, tg.Endpoint.ARROW))
    added += 1

params = tutil.Parameters()
params.set("coefLow", jpype.JDouble(args.coef_low))
params.set("coefHigh", jpype.JDouble(args.coef_high))
tutil.RandomUtil.getInstance().setSeed(jpype.JLong(args.seed))
im = tsem.SemIm(tsem.SemPm(lag_graph), params)

# Coefficient matrices B_k[child, parent] for k = 0..L, read off the SemIm for edges into lag-0 nodes.
name_to_index = {f"X{i + 1}": i for i in range(p)}
B = np.zeros((L + 1, p, p))
err_sd = np.ones(p)
true_into_lag0 = set()  # (parent_name_with_lag, child_base_name)
for node in lag_graph.getNodes():
    name = str(node.getName())
    if TsUtils.getLag(name) != 0:
        continue
    child = name_to_index[name]
    err_sd[child] = np.sqrt(float(im.getErrVar(node)))
    for parent in lag_graph.getParents(node):
        pname = str(parent.getName())
        lag = int(TsUtils.getLag(pname))
        par = name_to_index[str(TsUtils.getNameNoLag(pname))]
        B[lag, child, par] = float(im.getEdgeCoef(parent, node))
        true_into_lag0.add((pname if lag > 0 else pname, name))

n_true = len(true_into_lag0)
n_contemp_true = sum(1 for a, b in true_into_lag0 if TsUtils.getLag(a) == 0)
print(f"lag graph: p={p}, lags={L}, {lag_graph.getNumEdges()} edges total, {n_true} into lag 0 "
      f"({n_contemp_true} contemporaneous)")

def spectral_radius(B):
    Ainv = np.linalg.inv(np.eye(p) - B[0])
    comp = np.zeros((p * L, p * L))
    for k in range(1, L + 1):
        comp[:p, (k - 1) * p:k * p] = Ainv @ B[k]
    if L > 1:
        comp[p:, :-p] = np.eye(p * (L - 1))
    return max(abs(np.linalg.eigvals(comp)))

# A random lag graph with SemIm's coefficient range is often not a stationary VAR. Rather than simulate an
# exploding series (which yields a singular correlation matrix), shrink the LAGGED coefficients uniformly
# until the companion matrix is comfortably stable. Contemporaneous coefficients are left alone: they set
# the lag-0 DAG, not the dynamics.
rho = spectral_radius(B)
shrink = 1.0
while rho > 0.9:
    shrink *= 0.9
    B[1:] *= 0.9
    rho = spectral_radius(B)
print(f"VAR spectral radius {rho:.3f}" + (f" (lagged coefficients scaled by {shrink:.2f} for stationarity)" if shrink < 1 else ""))
Ainv = np.linalg.inv(np.eye(p) - B[0])

# --------------------------------------------------------------------------------------
# 2. Simulate the series
# --------------------------------------------------------------------------------------

rng = np.random.default_rng(args.seed)
burn = 500
T = args.T + burn
X = np.zeros((T, p))
for t in range(L, T):
    driver = np.zeros(p)
    for k in range(1, L + 1):
        driver += B[k] @ X[t - k]
    e = rng.normal(size=p) * err_sd
    X[t] = Ainv @ (driver + e)     # solves X_t = B0 X_t + driver + e
X = X[burn:]
series = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(p)])

# --------------------------------------------------------------------------------------
# 3. Lagged data and Tetrad's tiered knowledge via TsUtils
# --------------------------------------------------------------------------------------

def to_tetrad(df):
    vars_ = jutil.ArrayList()
    for c in df.columns:
        vars_.add(td.ContinuousVariable(str(c)))
    box = td.DoubleDataBox(df.shape[0], df.shape[1])
    arr = np.ascontiguousarray(df.values, dtype=np.float64)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            box.set(i, j, jpype.JDouble(arr[i, j]))
    return td.BoxDataSet(box, vars_)

def from_tetrad(ds):
    names = [str(v.getName()) for v in ds.getVariables()]
    m = np.array(ds.getDoubleData().toArray(), dtype=np.float64)
    return pd.DataFrame(m, columns=names)

lagged_ds = TsUtils.createLagData(to_tetrad(series), L)
knowledge = lagged_ds.getKnowledge()
lagged = from_tetrad(lagged_ds)
lag_names = list(lagged.columns)

# Tiers -> the dict C-BOSS takes. TsUtils puts lag L in tier 0 ... lag 0 in tier L, every variable in one.
tiers = {}
for t in range(knowledge.getNumTiers()):
    members = [str(s) for s in knowledge.getTier(t)]
    if members:
        tiers[t] = members
assert sorted(sum(tiers.values(), [])) == sorted(lag_names), "TsUtils knowledge should partition the lagged variables"
lag0_tier = max(tiers.keys())
forbid_within = {lag0_tier} if args.star else None
if args.star:
    knowledge.setTierForbiddenWithin(lag0_tier, True)

print(f"lagged data: {lagged.shape[0]} rows x {lagged.shape[1]} columns; tiers: "
      + ", ".join(f"tier {t}: {len(v)} vars (lag {L - t})" for t, v in sorted(tiers.items()))
      + (f"; lag-0 tier starred" if args.star else ""))

# --------------------------------------------------------------------------------------
# 4. Searches
# --------------------------------------------------------------------------------------

lag0_names = set(tiers[lag0_tier])

def edges_into_lag0_from_matrix(A, names):
    """A[c, pa] == 1 means pa -> c. Returns the set of (parent_name, child_name) with child at lag 0."""
    out = set()
    for c in range(len(names)):
        if names[c] not in lag0_names:
            continue
        for pa in np.nonzero(A[c])[0]:
            out.add((names[pa], names[c]))
    return out

def edges_into_lag0_from_graph(g):
    out = set()
    undirected = set()
    for e in g.getEdges():
        a, b = str(e.getNode1().getName()), str(e.getNode2().getName())
        e1, e2 = str(e.getEndpoint1().name()), str(e.getEndpoint2().name())
        if e2 == "ARROW" and e1 == "TAIL" and b in lag0_names:
            out.add((a, b))
        elif e1 == "ARROW" and e2 == "TAIL" and a in lag0_names:
            out.add((b, a))
        elif e1 == "TAIL" and e2 == "TAIL" and (a in lag0_names or b in lag0_names):
            undirected.add(frozenset((a, b)))
    return out, undirected

def score(est_directed, est_undirected=frozenset(), label=""):
    """Adjacency P/R over all edges into lag 0 (undirected counts as adjacency), arrowhead P/R over the
    lagged edges (which tiers orient) and, separately, over contemporaneous edges."""
    tru = true_into_lag0
    tru_adj = {frozenset(e) for e in tru}
    est_adj = {frozenset(e) for e in est_directed} | set(est_undirected)
    tp_adj = len(tru_adj & est_adj)
    ap = tp_adj / max(len(est_adj), 1)
    ar = tp_adj / max(len(tru_adj), 1)

    def arrows(edges, contemporaneous):
        return {e for e in edges if (TsUtils.getLag(e[0]) == 0) == contemporaneous}

    lines = [f"{label:28s} adjacency into lag 0: P={ap:.3f} R={ar:.3f}  (est {len(est_adj)}, true {len(tru_adj)})"]
    for contemp, tag in ((False, "lagged -> lag0"), (True, "lag0 -> lag0")):
        t, e = arrows(tru, contemp), arrows(est_directed, contemp)
        if len(t) == 0 and len(e) == 0:
            continue
        tp = len(t & e)
        lines.append(f"{'':28s}   {tag:15s} arrowheads: P={tp / max(len(e), 1):.3f} R={tp / max(len(t), 1):.3f}"
                     f"  (est {len(e)}, true {len(t)})")
    print("\n".join(lines))

# ---- C-BOSS with tiers ----
t0 = time.perf_counter()
A = cg.boss(lagged, discount=args.penalty, restarts=args.restarts, seed=args.seed,
            knowledge=tiers, forbid_within=forbid_within)
ms = (time.perf_counter() - t0) * 1000
A = np.asarray(A)
est = edges_into_lag0_from_matrix(A, lag_names)
violations = sum(1 for pa, c in est if TsUtils.getLag(pa) < TsUtils.getLag(c))  # later lag as parent of earlier: impossible under tiers
back_in_time = sum(1 for c in range(len(lag_names)) for pa in np.nonzero(A[c])[0]
                   if TsUtils.getLag(lag_names[pa]) < TsUtils.getLag(lag_names[c]))
print()
print(f"C-BOSS (tiers): {ms:.0f} ms, {int(A.sum())} edges in the full lagged DAG, "
      f"{back_in_time} edges pointing backward in time (must be 0)")
score(est, label="C-BOSS + tiers")

# ---- C-BOSS without knowledge, for contrast ----
if args.no_knowledge:
    A0 = np.asarray(cg.boss(lagged, discount=args.penalty, restarts=args.restarts, seed=args.seed))
    est0 = edges_into_lag0_from_matrix(A0, lag_names)
    back0 = sum(1 for c in range(len(lag_names)) for pa in np.nonzero(A0[c])[0]
                if TsUtils.getLag(lag_names[pa]) < TsUtils.getLag(lag_names[c]))
    print()
    print(f"C-BOSS (no knowledge): {int(A0.sum())} edges, {back0} pointing backward in time")
    score(est0, label="C-BOSS, no knowledge")

# ---- Tetrad BOSS with the same knowledge, as the reference ----
if args.tetrad_boss:
    sc = tscore.SemBicScore(lagged_ds, True)
    sc.setPenaltyDiscount(args.penalty)
    boss = ts.Boss(sc)
    boss.setNumStarts(args.restarts)
    boss.setUseBes(False)
    boss.setNumThreads(1)
    ps = ts.PermutationSearch(boss)
    ps.setKnowledge(knowledge)
    ps.setSeed(args.seed)
    t0 = time.perf_counter()
    g = ps.search()
    ms = (time.perf_counter() - t0) * 1000
    est_t, und_t = edges_into_lag0_from_graph(g)
    print()
    print(f"Tetrad BOSS (same knowledge): {ms:.0f} ms, {g.getNumEdges()} edges in the CPDAG")
    score(est_t, und_t, label="Tetrad BOSS + tiers")

    # agreement between the two on adjacencies into lag 0
    a_c = {frozenset(e) for e in est}
    a_t = {frozenset(e) for e in est_t} | set(und_t)
    print(f"\nC-BOSS vs Tetrad BOSS adjacencies into lag 0: {len(a_c & a_t)} shared, "
          f"{len(a_c - a_t)} only C-BOSS, {len(a_t - a_c)} only Tetrad")

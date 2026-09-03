"""
lagged_cboss.py -- run C-BOSS on a time series with lag-tier knowledge and report edges into lag 0.

    python lagged_cboss.py series.csv --lags 2 --penalty 2 --starts 5
    python lagged_cboss.py series.csv --lags 3 --penalty 3 --starts 10 --star --all-edges --out edges.txt

Input: a delimited file with a header row, one column per variable, one row per time step in order.
The delimiter is sniffed (comma, tab, semicolon, whitespace). Non-numeric columns are dropped with a
notice; rows with missing values are dropped after lagging.

Lagging follows Tetrad's convention: variable X at lag k is named "X:k" and holds the value k steps
earlier, so row t of the lagged data is (X_t, X_{t-1}, ..., X_{t-L}). The first L rows are lost.
Knowledge is the lag tiers -- lag L first, lag 0 last -- which C-BOSS requires to partition the
variables; they do. A variable at lag k may then only take parents from lags >= k, so no edge points
backward in time. --star additionally forbids contemporaneous (lag-0 -> lag-0) edges.

Output: every edge whose child is a lag-0 variable, "parent --> child", sorted by child then parent.
Lagged parents are oriented by the tiers; contemporaneous parents are oriented by the search and
should be read with the usual caution about score-equivalent orientations.
"""

import argparse
import csv
import sys

import numpy as np
import pandas as pd

import causalget as cg


def read_series(path):
    with open(path, "r", newline="") as f:
        sample = f.read(65536)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        sep = dialect.delimiter
    except csv.Error:
        sep = r"\s+"
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    numeric = df.select_dtypes(include=[np.number])
    dropped = [c for c in df.columns if c not in numeric.columns]
    if dropped:
        print(f"note: dropping {len(dropped)} non-numeric column(s): {dropped[:8]}{' ...' if len(dropped) > 8 else ''}",
              file=sys.stderr)
    if numeric.shape[1] < 2:
        sys.exit("need at least two numeric columns")
    for c in numeric.columns:
        if ":" in c:
            sys.exit(f"column name {c!r} contains ':', which the lag naming uses; please rename it")
    return numeric


def lag_data(df, num_lags):
    """Tetrad's createLagData in pandas: X:k is X shifted forward by k steps."""
    blocks = []
    for lag in range(num_lags + 1):
        shifted = df.shift(lag)
        shifted.columns = [c if lag == 0 else f"{c}:{lag}" for c in df.columns]
        blocks.append(shifted)
    lagged = pd.concat(blocks, axis=1).iloc[num_lags:]
    before = len(lagged)
    lagged = lagged.dropna()
    if len(lagged) < before:
        print(f"note: dropped {before - len(lagged)} row(s) with missing values", file=sys.stderr)
    return lagged.reset_index(drop=True)


def lag_of(name):
    return int(name.rsplit(":", 1)[1]) if ":" in name else 0


def main():
    ap = argparse.ArgumentParser(description="C-BOSS on lagged time-series data with lag-tier knowledge.")
    ap.add_argument("csv", help="delimited file with a header; one column per variable, rows in time order")
    ap.add_argument("--lags", type=int, required=True, help="number of lags to include")
    ap.add_argument("--penalty", type=float, default=2.0, help="BIC penalty discount")
    ap.add_argument("--starts", type=int, default=1, help="number of random restarts (local searches)")
    ap.add_argument("--star", action="store_true", help="forbid contemporaneous (lag-0 -> lag-0) edges")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--tol", type=float, default=1e-2, help="min improvement in BIC points to accept a move")
    ap.add_argument("--all-edges", action="store_true", help="also print the full lagged DAG")
    ap.add_argument("--out", type=str, default=None, help="write the lag-0 edges to this file (one 'parent --> child' per line)")
    args = ap.parse_args()
    if args.lags < 1:
        sys.exit("--lags must be at least 1")

    series = read_series(args.csv)
    lagged = lag_data(series, args.lags)
    names = list(lagged.columns)
    p0 = series.shape[1]
    print(f"{args.csv}: {series.shape[0]} time steps x {p0} variables -> lagged: {lagged.shape[0]} rows x {lagged.shape[1]} columns "
          f"(lags 0..{args.lags})")

    # Tiers: lag L first ... lag 0 last. Every lagged variable is in exactly one.
    tiers = {t: [c for c in names if lag_of(c) == args.lags - t] for t in range(args.lags + 1)}
    lag0_tier = args.lags
    forbid_within = {lag0_tier} if args.star else None

    R = lagged.corr().values
    if not np.isfinite(R).all():
        sys.exit("the lagged correlation matrix has NaN entries (a constant column?)")

    A = np.asarray(cg.boss(lagged, discount=args.penalty, restarts=args.starts, seed=args.seed, tol=args.tol,
                           knowledge=tiers, forbid_within=forbid_within))

    into_lag0 = []
    backward = 0
    for c in range(len(names)):
        for pa in np.nonzero(A[c])[0]:
            if lag_of(names[pa]) < lag_of(names[c]):
                backward += 1
            if lag_of(names[c]) == 0:
                into_lag0.append((names[pa], names[c]))
    into_lag0.sort(key=lambda e: (e[1], lag_of(e[0]), e[0]))

    n_lagged = sum(1 for pa, _ in into_lag0 if lag_of(pa) > 0)
    n_contemp = len(into_lag0) - n_lagged
    print(f"C-BOSS: penalty {args.penalty:g}, {args.starts} start(s), {int(A.sum())} edges in the lagged DAG, "
          f"{backward} pointing backward in time" + ("" if backward == 0 else "  <-- should be 0; please report"))
    print(f"edges into lag 0: {len(into_lag0)}  ({n_lagged} from lagged variables, {n_contemp} contemporaneous"
          + (", contemporaneous forbidden" if args.star else "") + ")")
    print()
    current_child = None
    for pa, ch in into_lag0:
        if ch != current_child:
            current_child = ch
            parents = [q for q, cc in into_lag0 if cc == ch]
            print(f"{ch}  <--  {', '.join(parents)}")
    print()
    for pa, ch in into_lag0:
        print(f"{pa} --> {ch}")

    if args.all_edges:
        print("\nfull lagged DAG:")
        for c in range(len(names)):
            for pa in np.nonzero(A[c])[0]:
                print(f"{names[pa]} --> {names[c]}")

    if args.out:
        with open(args.out, "w") as f:
            for pa, ch in into_lag0:
                f.write(f"{pa} --> {ch}\n")
        print(f"\nwrote {len(into_lag0)} edges to {args.out}")


if __name__ == "__main__":
    main()
